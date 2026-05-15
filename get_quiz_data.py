#!/usr/bin/env python3
"""
Script to retrieve quiz/assessment data using UUID from various APIs

Usage:
  python get_quiz_data.py <uuid_or_url> [options]

Examples:
  python get_quiz_data.py 956626a8-5d53-43a2-a195-06355757d8cb --external
  python get_quiz_data.py https://assessments.botgo.io/assessments/956626a8-5d53-43a2-a195-06355757d8cb --external
  python get_quiz_data.py assessments.botgo.io/api/assessment/info/46ece8ee-073f-4a12-80b0-b4188c587ee8 --info4000
"""

import sys
import re
import json
import requests


class QuizDataRetriever:
    def __init__(
        self,
        base_url="https://assessments.botgo.io",
        external_api_url="https://assessments.botgo.io",
        info_api_url="https://assessments.botgo.io",
    ):
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api"

        # Botgo panel (port 3000)
        self.external_api_url = external_api_url.rstrip('/')
        self.assessments_endpoint = f"{self.external_api_url}/assessments"

        # Info API (port 4000)
        self.info_api_url = info_api_url.rstrip('/')
        self.info_endpoint = f"{self.info_api_url}/api/assessment/info"

    def extract_uuid(self, input_str):
        """Extract UUID from URL or return UUID if already provided"""
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        match = re.search(uuid_pattern, input_str, re.IGNORECASE)
        if match:
            return match.group(0)
        raise ValueError(f"No valid UUID found in: {input_str}")

    def get_assessment_data(self, uuid):
        """Fetch assessment data from Botgo panel (assessments.botgo.io)"""
        url = f"{self.assessments_endpoint}/{uuid}"
        try:
            print(f"Fetching assessment data from: {url}")
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                json_response = response.json()

                if isinstance(json_response, dict) and 'success' in json_response:
                    if json_response.get('success'):
                        return json_response.get('data', json_response)
                    else:
                        print("API returned success=false")
                        print(f"Error: {json_response.get('error', 'Unknown error')}")
                        return None

                return json_response

            elif response.status_code == 404:
                print(f"Assessment not found: {uuid}")
                return None
            else:
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            print(f"Connection failed. Is the Botgo panel running at {self.external_api_url}?")
            print("Make sure the assessments.botgo.io is accessible.")
            return None
        except requests.exceptions.Timeout:
            print("Request timed out")
            return None
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def get_assessment_info_4000(self, uuid):
        """Fetch assessment info from assessments.botgo.io/api/assessment/info/<uuid>"""
        url = f"{self.info_endpoint}/{uuid}"
        try:
            print(f"Fetching assessment info from: {url}")
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                json_response = response.json()

                # Handle { "success": true, "data": {...} }
                if isinstance(json_response, dict) and 'success' in json_response:
                    if json_response.get('success'):
                        return json_response.get('data', json_response)
                    else:
                        print("Info API returned success=false")
                        print(f"Error: {json_response.get('error', 'Unknown error')}")
                        return None

                return json_response

            elif response.status_code == 404:
                print(f"Assessment info not found: {uuid}")
                return None
            else:
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            print(f"Connection failed. Is the info API running at {self.info_api_url}?")
            print("Make sure the assessments.botgo.io is accessible.")
            return None
        except requests.exceptions.Timeout:
            print("Request timed out")
            return None
        except Exception as e:
            print(f"Error fetching info data: {e}")
            return None

    def get_quiz_config(self, quiz_id):
        """Fetch quiz configuration from API (assessments.botgo.io)"""
        url = f"{self.api_base}/config/{quiz_id}"
        try:
            print(f"Fetching quiz data from: {url}")
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                print(f"Quiz not found: {quiz_id}")
                return None
            else:
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            print(f"Connection failed. Is the server running at {self.base_url}?")
            return None
        except requests.exceptions.Timeout:
            print("Request timed out")
            return None
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def get_all_quizzes(self):
        """Fetch all available quizzes from local API"""
        url = f"{self.api_base}/config"
        try:
            print(f"Fetching all quizzes from: {url}")
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: {response.status_code}")
                return None

        except Exception as e:
            print(f"Error fetching quizzes: {e}")
            return None

    def check_health(self):
        """Check API health status (assessments.botgo.io)"""
        url = f"{self.api_base}/health"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print("API is healthy")
                openai_info = health_data.get('openai', {})
                print(f"OpenAI Client: {'Available' if openai_info.get('client_available') else 'Not Available'}")
                print(f"API Key Set: {'Yes' if openai_info.get('api_key_set') else 'No'}")
                return True
            return False
        except Exception:
            print("API is not responding")
            return False

    def display_quiz_data(self, quiz_data):
        """Display quiz data from local API in a readable format"""
        if not quiz_data:
            return

        print("\n" + "=" * 60)
        print("QUIZ INFORMATION")
        print("=" * 60)

        print(f"\nTitle: {quiz_data.get('title', 'N/A')}")
        print(f"ID: {quiz_data.get('id', 'N/A')}")
        print(f"Description: {quiz_data.get('description', 'N/A')}")

        if 'questions' in quiz_data:
            questions = quiz_data['questions']
            print(f"\nTotal Questions: {len(questions)}")
            print("\nQuestions:")
            for i, q in enumerate(questions, 1):
                print(f"\n  {i}. {q.get('text', 'N/A')}")
                if 'options' in q:
                    for opt in q['options']:
                        print(f"     - {opt.get('text', 'N/A')} (Score: {opt.get('score', 0)})")

        print("\n" + "=" * 60)

    def display_assessment_data(self, assessment_data):
        """Display assessment data from Botgo panel in a readable format"""
        if not assessment_data:
            return

        print("\n" + "=" * 70)
        print("BOTGO ASSESSMENT DATA")
        print("=" * 70)

        print(f"\nID: {assessment_data.get('id', 'N/A')}")
        print(f"UUID: {assessment_data.get('uuid', 'N/A')}")
        print(f"BG Account ID: {assessment_data.get('bgAccountId', 'N/A')}")

        print(f"\nTitle: {assessment_data.get('title', 'N/A')}")
        print(f"Subtitle: {assessment_data.get('subtitle', 'N/A')}")
        print(f"Description: {assessment_data.get('description', 'N/A')}")

        print(f"\nCategory: {assessment_data.get('category', 'N/A')}")
        print(f"Icon: {assessment_data.get('icon', 'N/A')}")
        print(f"Color: {assessment_data.get('color', 'N/A')}")

        print(f"\nStatus: {assessment_data.get('status', 'N/A')}")
        print(f"Created: {assessment_data.get('createdAt', 'N/A')}")
        print(f"Updated: {assessment_data.get('updatedAt', 'N/A')}")

        total_marks = assessment_data.get('totalMarks')
        passing_marks = assessment_data.get('passingMarks')
        duration = assessment_data.get('durationMinutes')

        if total_marks is not None:
            print(f"\nTotal Marks: {total_marks}")
        if passing_marks is not None:
            print(f"Passing Marks: {passing_marks}")
        if duration is not None:
            print(f"Duration: {duration} minutes")

        questions = assessment_data.get('questions', [])
        if questions:
            print(f"\nQuestions: {len(questions)} question(s)")
            for i, question in enumerate(questions, 1):
                print(f"\n   {i}. {question}")
        else:
            print("\nQuestions: No questions available")

        survey_forms = assessment_data.get('surveyForms')
        if survey_forms is not None:
            print(f"\nSurvey Forms: {survey_forms}")

        responses = assessment_data.get('responses')
        if responses is not None:
            print(f"\nResponses: {responses}")

        shareable_link = assessment_data.get('shareableLink')
        if shareable_link:
            print(f"\nShareable Link: {shareable_link}")

        print("\n" + "=" * 70)

    def display_assessment_info_4000(self, data):
        """Display assessment info from 4000 API"""
        if not data:
            return

        print("\n" + "=" * 70)
        print("ASSESSMENT INFO (4000)")
        print("=" * 70)

        print(f"\nID: {data.get('id', 'N/A')}")
        print(f"UUID: {data.get('uuid', 'N/A')}")
        print(f"BG Account ID: {data.get('bgAccountId', 'N/A')}")

        print(f"\nTitle: {data.get('title', 'N/A')}")
        print(f"Subtitle: {data.get('subtitle', 'N/A')}")
        print(f"Description: {data.get('description', 'N/A')}")

        print(f"\nCategory: {data.get('category', 'N/A')}")
        print(f"Icon: {data.get('icon', 'N/A')}")
        print(f"Color: {data.get('color', 'N/A')}")

        print(f"\nStatus: {data.get('status', 'N/A')}")
        print(f"Created: {data.get('createdAt', 'N/A')}")
        print(f"Updated: {data.get('updatedAt', 'N/A')}")

        total_marks = data.get('totalMarks')
        passing_marks = data.get('passingMarks')
        duration = data.get('durationMinutes')

        if total_marks is not None:
            print(f"\nTotal Marks: {total_marks}")
        if passing_marks is not None:
            print(f"Passing Marks: {passing_marks}")
        if duration is not None:
            print(f"Duration: {duration} minutes")

        questions = data.get('questions', [])
        if questions:
            print(f"\nQuestions: {len(questions)} question(s)")
            for i, question in enumerate(questions, 1):
                print(f"\n   {i}. {question}")
        else:
            print("\nQuestions: No questions available")

        survey_forms = data.get('surveyForms')
        if survey_forms is not None:
            print(f"\nSurvey Forms: {survey_forms}")

        responses = data.get('responses')
        if responses is not None:
            print(f"\nResponses: {responses}")

        shareable_link = data.get('shareableLink')
        if shareable_link:
            print(f"\nShareable Link: {shareable_link}")

        print("\n" + "=" * 70)

    def display_json_pretty(self, data):
        """Display data as raw JSON (single line)"""
        if not data:
            return
        print(json.dumps(data, ensure_ascii=False))

    def save_to_file(self, quiz_data, filename=None):
        """Save data to JSON file"""
        if not quiz_data:
            return

        if not filename:
            quiz_id = quiz_data.get('id', 'quiz')
            filename = f"{quiz_id}_data.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(quiz_data, f, indent=2, ensure_ascii=False)
            print(f"Data saved to: {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")


def main():
    """Main execution function"""

    if len(sys.argv) < 2:
        print("Usage: python get_quiz_data.py <uuid_or_url>")
        print("\nExamples:")
        print("  python get_quiz_data.py 956626a8-5d53-43a2-a195-06355757d8cb --external")
        print("  python get_quiz_data.py https://assessments.botgo.io/assessments/956626a8-5d53-43a2-a195-06355757d8cb --external")
        print("  python get_quiz_data.py assessments.botgo.io/api/assessment/info/46ece8ee-073f-4a12-80b0-b4188c587ee8 --info4000")
        print("\nOptions:")
        print("  --external      Fetch from Botgo panel (assessments.botgo.io)")
        print("  --info4000      Fetch from info API (assessments.botgo.io/api/assessment/info/<uuid>)")
        print("  --local         Fetch from API (assessments.botgo.io) [default]")
        print("  --json          Display raw JSON output")
        print("  --save          Save data to JSON file")
        print("  --all           List all available quizzes")
        print("  --health        Check API health status (assessments.botgo.io)")
        sys.exit(1)

    retriever = QuizDataRetriever()

    if '--health' in sys.argv:
        retriever.check_health()
        return

    if '--all' in sys.argv:
        quizzes = retriever.get_all_quizzes()
        if quizzes:
            print("\nAvailable Quizzes:")
            for quiz in quizzes:
                print(f"\n  - {quiz.get('title', 'N/A')}")
                print(f"    ID: {quiz.get('id', 'N/A')}")
                desc = quiz.get('description', '')
                if desc:
                    print(f"    Description: {desc[:100]}...")
        return

    input_str = sys.argv[1]

    try:
        uuid = retriever.extract_uuid(input_str)
        print(f"Extracted UUID: {uuid}")

        use_external = '--external' in sys.argv
        use_info4000 = '--info4000' in sys.argv
        show_json = '--json' in sys.argv

        if use_info4000:
            print(f"\nUsing info API: {retriever.info_api_url}")
            print("\nRetrieving assessment info data...")
            info_data = retriever.get_assessment_info_4000(uuid)

            if info_data:
                if show_json:
                    retriever.display_json_pretty(info_data)
                else:
                    retriever.display_assessment_info_4000(info_data)

                if '--save' in sys.argv:
                    retriever.save_to_file(info_data, f"assessment_info_{uuid}.json")

            else:
                print("\nFailed to retrieve assessment info data")
                print("\nTips:")
                print("  - Make sure assessments.botgo.io is accessible")
                print("  - Verify the UUID is correct")
                sys.exit(1)

        elif use_external:
            print(f"\nUsing Botgo Panel API: {retriever.external_api_url}")
            print("\nRetrieving assessment data...")
            assessment_data = retriever.get_assessment_data(uuid)

            if assessment_data:
                if show_json:
                    retriever.display_json_pretty(assessment_data)
                else:
                    retriever.display_assessment_data(assessment_data)

                if '--save' in sys.argv:
                    retriever.save_to_file(assessment_data, f"assessment_{uuid}.json")

            else:
                print("\nFailed to retrieve assessment data")
                print("\nTips:")
                print("  - Make sure assessments.botgo.io is accessible")
                print("  - Verify the UUID is correct")
                print("  - Check if the assessment exists in the panel")
                sys.exit(1)

        else:
            print(f"\nUsing local API: {retriever.base_url}")
            print("\nChecking API health...")
            retriever.check_health()

            print("\nRetrieving quiz data...")
            quiz_data = retriever.get_quiz_config(uuid)

            if quiz_data:
                if show_json:
                    retriever.display_json_pretty(quiz_data)
                else:
                    retriever.display_quiz_data(quiz_data)

                if '--save' in sys.argv:
                    retriever.save_to_file(quiz_data)

            else:
                print("\nFailed to retrieve quiz data")
                print("\nTip: Try using --external or --info4000 depending on where your data lives")
                sys.exit(1)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
