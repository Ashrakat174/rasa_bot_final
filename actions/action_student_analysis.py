import os
import sqlite3
import json
import requests
import logging
from dotenv import load_dotenv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import SlotSet

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
DB_PATH = "actions/academic_advisor.db"

class ActionLLMAcademicPlan(Action):
    def name(self):
        return "action_student_analysis"


    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict):
        # استخدم الرسالة الأخيرة + القيم المخزنة في الـ slots
        user_input = tracker.latest_message.get("text", "")
        student_new = self.extract_student_info(user_input)
        if student_new is None:
            # إذا لم يرجع الذكاء الاصطناعي JSON، احتفظ بالبيانات القديمة فقط
            student_new = {}
            logger.debug("[student extracted] No valid JSON, using only slots.")
        else:
            logger.debug(f"[student extracted] {student_new}")

        # تحديث تدريجي لكل حقل: إذا أرسل المستخدم قيمة جديدة لأي حقل، يتم تحديثه فورًا في الـ slots
        slot_keys = [
            ("name", "student_name"),
            ("level", "level"),
            ("term", "term"),
            ("gpa", "gpa"),
            ("department", "department"),
            ("failed_courses", "removed_subjects"),
            ("passed_courses", "passed_courses")
        ]
        student = {}
        # اجمع بيانات الـ slots القديمة أولاً
        for key, slot_name in slot_keys:
            val = tracker.get_slot(slot_name)
            if val not in [None, "", [], "null"]:
                student[key] = val
        # ثم حدث أي قيمة جديدة أرسلها المستخدم (ولو جزئية)
        for key, slot_name in slot_keys:
            if key in student_new and student_new[key] not in [None, "", [], "null"]:
                # دمج المواد الراسبة والمجتازة
                if key == "failed_courses":
                    old_failed = student.get("failed_courses", [])
                    merged_failed = list(set([str(f).strip().lower() for f in (student_new["failed_courses"] + old_failed) if f]))
                    student["failed_courses"] = merged_failed
                elif key == "passed_courses":
                    old_passed = student.get("passed_courses", [])
                    merged_passed = list(set([str(f).strip().lower() for f in (student_new["passed_courses"] + old_passed) if f]))
                    student["passed_courses"] = merged_passed
                else:
                    student[key] = student_new[key]

        # تحقق من اكتمال الحقول الأساسية
        required_fields = ["gpa", "level", "term", "department"]
        missing = []
        for field in required_fields:
            value = student.get(field)
            if value is None or (isinstance(value, str) and not value) or (isinstance(value, list) and not value):
                missing.append(field)

        if missing:
            fields_ar = {
                "gpa": "المعدل التراكمي (gpa)",
                "level": "المستوى الدراسي",
                "term": "الترم",
                "department": "القسم"
            }
            missing_fields = ', '.join([fields_ar.get(f, f) for f in missing])
            dispatcher.utter_message(f"⚠️ يرجى إدخال البيانات التالية: {missing_fields}")
            logger.debug(f"[missing fields] {missing}, [student] {student}")
            return []

        # إذا لم تُذكر المواد الراسبة أو المجتازة، اعتبرها قائمة فارغة
        if student.get("failed_courses") is None:
            student["failed_courses"] = []
        if student.get("passed_courses") is None:
            student["passed_courses"] = []

        all_courses = self.load_courses_from_db()

        try:
            level = int(student.get("level", 0))
        except Exception:
            level = 0
        try:
            term = int(student.get("term", 1))
        except Exception:
            term = 1
        try:
            gpa = float(str(student.get("gpa")).replace("٫", ".").replace(",", ".")) or 0.0
        except Exception:
            gpa = 0.0

        failed_courses = student.get("failed_courses") or []
        failed = [self.normalize_arabic_text(f) for f in failed_courses]
        passed_courses = student.get("passed_courses") or []
        passed = [self.normalize_arabic_text(p) for p in passed_courses]
        raw_department = self.normalize_arabic_text((student.get("department") or ""))
        department = self.normalize_department(raw_department)

        # ترشيح المواد: فقط من المستوى الحالى أو المستويات القادمة
        # إذا كان هناك مادة رسب فيها الطالب ومفتوحة فى الترم الحالى (المنتقل إليه)، يتم ترشيحها حتى لو كانت من مستوى أقل
        filtered_courses = []
        for course in all_courses:
            cname = self.normalize_arabic_text(course["name"])
            cterm = course["term"]
            clevel = course["level"]
            ctype = course["type"].strip().lower()
            cdept = [self.normalize_department(self.normalize_arabic_text(d)) for d in course["department"]]
            prereqs = [self.normalize_arabic_text(p) for p in course["prerequisites"] or []]

            logger.debug(f"[filter] المادة: {cname} | المستوى: {clevel} | الترم: {cterm} | مستوى الطالب: {level} | الترم الحالي: {term}")

            if not cname:
                continue
            if ctype != "اجباري":
                continue
            if cterm != term:
                continue
            if not any(department in d or "كل" in d for d in cdept):
                continue
            # لا ترشح المادة إذا كانت مجتازة
            if cname in passed:
                continue
            # إذا كانت المادة راسب فيها الطالب ومفتوحة فى الترم الحالى، يتم ترشيحها حتى لو كانت من مستوى أقل
            if cname in failed:
                if cterm == term:
                    if not self.is_prereq_satisfied(prereqs, failed, passed, all_courses, level):
                        continue
                    filtered_courses.append(course)
                continue  # لا ترشح المواد الراسبة من مستويات أقل من مستوى الطالب إلا إذا كانت مفتوحة فى الترم الحالى
            # المواد الجديدة (لم يرسب ولم ينجح فيها)
            if clevel < level:
                continue
            if not self.is_prereq_satisfied(prereqs, failed, passed, all_courses, level):
                continue
            filtered_courses.append(course)

        # حساب الحد الأقصى للساعات بناءً على المعدل التراكمي
        max_hours = 18 if gpa >= 2 else 15 if gpa >= 1 else 9
        # اختيار مجموعة مواد بحيث يكون مجموع الساعات = الحد الأقصى (وليس عدد المواد)
        from itertools import combinations
        best_combo = []
        best_hours = 0
        sorted_courses = sorted(filtered_courses, key=lambda x: (x["level"], x["name"]))
        n = len(sorted_courses)
        # جرب كل التركيبات الممكنة حتى تجد مجموعة ساعاتها = الحد الأقصى أو أقرب قيمة أقل
        for r in range(n, 0, -1):
            for combo in combinations(sorted_courses, r):
                hours = sum(c.get("credit_hours", 0) for c in combo)
                if hours == max_hours:
                    best_combo = list(combo)
                    best_hours = hours
                    break
                elif hours < max_hours and hours > best_hours:
                    best_combo = list(combo)
                    best_hours = hours
            if best_hours == max_hours:
                break
        selected_courses = best_combo
        total_hours = best_hours
        logger.debug(f"[selected_courses] {[c['name'] for c in selected_courses]} | total_hours={total_hours}")

        messages = self.build_prompt(student, selected_courses, batch_id=1)
        response = self.ask_gpt(messages)
        dispatcher.utter_message(response.strip())

        # تحديث جميع الـ slots بنفس الأسماء في domain.yml
        slot_events = []
        def safe_slot(key, slot_name, default=None):
            val = student.get(key, default)
            if val not in [None, "", [], "null"]:
                slot_events.append(SlotSet(slot_name, val))

        for key, slot_name in slot_keys:
            safe_slot(key, slot_name)

        slot_events.append(SlotSet("removed_subjects", student.get("failed_courses", [])))
        slot_events.append(SlotSet("passed_courses", student.get("passed_courses", [])))

        return slot_events

    # تم حذف منطق جمع الرسائل، نستخدم فقط latest_message

    def is_prereq_satisfied(self, prereqs, failed_courses, passed_courses, all_courses, student_level):
        failed = [self.normalize_arabic_text(f) for f in failed_courses]
        passed = [self.normalize_arabic_text(p) for p in passed_courses]
        for prereq in prereqs:
            prereq_name = self.normalize_arabic_text(prereq)
            if prereq_name in failed:
                return False
            if prereq_name in passed:
                continue
            for course in all_courses:
                course_name_norm = self.normalize_arabic_text(course["name"])
                if course_name_norm == prereq_name:
                    if course["level"] < student_level:
                        return True
                    else:
                        return prereq_name not in failed
        return True

    def normalize_department(self, name):
        replacements = {
            "الاله": "الآلة",
            "ذكاء الاله": "ذكاء الآلة",
            "ذكاء الآله": "ذكاء الآلة",
            "علوم الحاسب": "علوم الحاسب",
            "نظم ذكيه": "نظم المعلومات",
            "علوم بيانات": "علم البيانات"
        }
        for wrong, correct in replacements.items():
            if wrong in name:
                return correct
        return name

    def extract_student_info(self, text):
        try:
            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "أنت مساعد أكاديمي ذكي. مهمتك الوحيدة استخراج بيانات الطالب التالية من أي وصف أو صياغة أو ترتيب أو لغة يكتبها المستخدم، حتى لو كانت غير مرتبة أو فيها أخطاء إملائية أو اختصارات. يجب أن يكون الرد JSON فقط وبدون أي شرح أو نص إضافي أو تعليقات أو رموز تنسيق. يجب أن ترجع جميع الحقول المطلوبة دومًا (name, level, term, gpa, failed_courses, department)، وإذا لم تتوفر معلومة أرجعها كقيمة null أو قائمة فارغة. لا تكتب أي شيء آخر غير JSON."
                            )
                        },
                        {"role": "user", "content": text}
                    ]
                }
            )
            logger.debug(f"[OpenRouter status] {res.status_code}")
            logger.debug(f"[OpenRouter content] {res.text}")
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"]
                logger.debug(f"[extract_student_info raw content] {content}")
                # تنظيف: إزالة أي سطر في البداية فيه فقط json أو ```json
                import re
                cleaned = content.strip()
                # أزل أي أسطر تبدأ بـ ``` أو json أو ```json
                cleaned = re.sub(r"^(json|```json|```)[\s\n]*", "", cleaned, flags=re.IGNORECASE)
                # أزل أي ``` في النهاية
                cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
                # محاولة التحويل مباشرة
                try:
                    data = json.loads(cleaned)
                except Exception as e2:
                    logger.error(f"[extract_student_info error] Response is not JSON after cleaning: {cleaned}")
                    return None
                # معالجة null في failed_courses
                if "failed_courses" in data and (data["failed_courses"] is None or data["failed_courses"] == "null"):
                    data["failed_courses"] = []
                return data
            else:
                logger.error(f"[extract_student_info error] API status: {res.status_code}, content: {res.text}")
                return None
        except Exception as e:
            logger.error(f"[extract_student_info error] {str(e)}")
            return None

    def load_courses_from_db(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT code, name, credit_hours, lecture_hours, lab_hours, prereq_id, prerequisites, department, type, level, term FROM المقررات")
        rows = cursor.fetchall()
        conn.close()

        courses = []
        for row in rows:
            try:
                prereqs = json.loads(row[6]) if row[6] else []
            except:
                prereqs = [row[6]] if row[6] else []
            courses.append({
                "code": row[0] or "",
                "name": row[1] or "",
                "credit_hours": row[2] or 0,
                "lecture_hours": row[3] or 0,
                "lab_hours": row[4] or 0,
                "prereq_id": row[5] or "",
                "prerequisites": prereqs,
                "department": [d.strip() for d in (row[7] or "كل الاقسام").split(",")],
                "type": row[8] or "",
                "level": row[9] or 0,
                "term": row[10] or 0
            })
        return courses

    def build_prompt(self, student, courses, batch_id=1):
        return [
            {
                "role": "system",
                "content": f"""أنت مساعد أكاديمي ذكي. إليك بيانات طالب وبعض المقررات (الدفعة {batch_id}).

مهمتك:
1. تحليل حالة الطالب الدراسية (المستوى، الترم، القسم، المعدل، المواد الراسبة).
2. قائمة المواد في الأسفل تمثل كل المواد **المتاحة للتسجيل** لهذا الطالب، حسب الشروط التالية:
- يجب أن تكون من الترم الحالي فقط.
- لا تشمل مواد نجح فيها الطالب سابقًا.
- لا تشمل مواد لها متطلب سابق (prerequisite) رسب فيه الطالب.
- لا تشمل مواد اختيارية.
- فقط المواد التي تنتمي لنفس القسم أو لجميع الأقسام.

✴ المطلوب منك:
- اختر أكبر عدد من هذه المواد للتسجيل، بشرط ألا يتجاوز مجموع الساعات الحد المسموح به.
- احسب الحد الأقصى للساعات كالتالي:
  - المعدل < 1 → 9 ساعات
  - المعدل بين 1 و 2 → 15 ساعة
  - المعدل ≥ 2 → 18 ساعة

📌 أعد الرد بالتنسيق التالي فقط:
📋 المواد المرشحة للتسجيل:
- اسم المادة (عدد الساعات)

🧮 عدد المواد المؤهلة: {len(courses)}
📦 إجمالي عدد الساعات المرشحة: [يحسبه النموذج]
🎯 الحد الأقصى المسموح به حسب المعدل: [يحسبه النموذج]

💡 ملاحظات:
- يجب تضمين المواد الراسبة إن كانت مؤهلة.
- لا تضف أي شروحات إضافية أو توجيهات.
- اضافة تنبيهات للطالب إذا كان هناك مواد راسبة أو متطلبات سابقة غير مستوفاة.
- اضافة تنبيه ايضا انه يجب ان ينتبه الى الجدول الدراسى وعدم تعارض المواعيد.
- اضافة تنبيه للطالب بضرورة مراجعة المرشد الأكاديمي قبل التسجيل.
"""
            },
            {
                "role": "user",
                "content": json.dumps({
                    "student": student,
                    "courses": courses
                }, ensure_ascii=False, default=str)
            }
        ]

    def ask_gpt(self, messages):
        try:
            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": messages,
                    "max_tokens": 1200
                } 
            )
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
            else:
                return f"❌ حدث خطأ أثناء الاتصال بـ GPT: {res.status_code}"
        except Exception as e:
            return f"❌ خطأ أثناء الاتصال بـ GPT: {str(e)}"

    def normalize_arabic_text(self, text):
        """
        توحيد الحروف العربية الشائعة التي تسبب لخبطة في المطابقة (الهاء/ة، الألف/إ/آ/ا، الياء/ى/ي، إلخ)
        """
        if not isinstance(text, str):
            return text
        replacements = {
            "ة": "ه",
            "ه": "ه",
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ى": "ي",
            "ئ": "ي",
            "ؤ": "و",
            "ٱ": "ا",
            "ﻯ": "ي",
            "ﻱ": "ي",
            "ﻲ": "ي",
            "ﻰ": "ي",
            "ـ": "",
            "  ": " ",
        }
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        return text.strip().lower()