"""
Flask application for Bromho Assessments - LLM Gateway
Provides API endpoints for quiz configuration and AI analysis using OpenAI
"""
import os
import json
import logging
import smtplib
import tempfile
import re
import random
import math
import urllib.request
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from io import BytesIO
from flask import Flask, jsonify, request, send_from_directory, Response, Blueprint
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
env_example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env.example')
log_temp = logging.getLogger(__name__)

# Check if .env file exists
if not os.path.exists(env_path):
    log_temp.warning('=' * 60)
    log_temp.warning('⚠ .env file not found!')
    log_temp.warning('To set up your environment:')
    log_temp.warning('  1. Copy env.example to .env:')  
    log_temp.warning('     Windows: copy env.example .env')
    log_temp.warning('     Linux/Mac: cp env.example .env')
    log_temp.warning('  2. Edit .env and add your OPENAI_API_KEY')
    log_temp.warning('=' * 60)

env_loaded = load_dotenv(env_path)
if env_loaded:
    log_temp.info('✓ Environment variables loaded from .env file')
elif os.path.exists(env_path):
    log_temp.info('Environment file exists but no new variables loaded')
else:
    log_temp.warning('No .env file found - using environment variables only')

# Initialize Flask app
app = Flask(__name__, static_folder='public', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Create Blueprint with URL prefix
bp = Blueprint('main', __name__, url_prefix='/maturity-assessments')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# Try to import BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    log.warning('BeautifulSoup4 not available. HTML parsing will use regex fallback.')

# Configuration
PORT = int(os.getenv('PORT', 7000))
QUIZZES_PATH = os.path.join('public', 'config', 'quizzes.json')
PROMPTS_PATH = os.path.join('public', 'config', 'prompt.json')
LOG_FILE = 'gateway.log'

# Initialize OpenAI client
openai_client = None
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_init_error = None

def initialize_openai_client():
    """Initialize OpenAI client with proper error handling for modern openai library"""
    global openai_client, openai_init_error, openai_api_key
    
    # Re-read API key in case it wasn't loaded initially
    if not openai_api_key:
        openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        log.warning('⚠ OPENAI_API_KEY not found in environment variables. Please check your .env file')
        return False
    
    # Check if key looks valid (starts with 'sk-' for OpenAI keys)
    if not openai_api_key.strip() or not openai_api_key.strip().startswith('sk-'):
        log.warning('⚠ OPENAI_API_KEY appears to be invalid. Please set a valid API key in .env file')
        return False
    
    # Temporarily remove proxy environment variables to prevent httpx issues
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy']
    saved_proxies = {}
    
    for var in proxy_vars:
        if var in os.environ:
            saved_proxies[var] = os.environ.pop(var)
    
    def restore_proxies():
        """Restore proxy environment variables"""
        for var, value in saved_proxies.items():
            os.environ[var] = value
    
    try:
        # Modern OpenAI library (>=1.0.0) - simple initialization
        openai_client = OpenAI(api_key=openai_api_key)
        
        # Restore proxy env vars
        restore_proxies()
        
        if saved_proxies:
            log.info('✓ OpenAI client initialized (proxy environment variables handled)')
        else:
            log.info('✓ OpenAI client initialized successfully')
        
        openai_init_error = None
        return True
        
    except TypeError as e:
        error_msg = str(e)
        
        # If there's a proxy/httpx issue, try with custom httpx client
        if 'proxies' in error_msg or 'unexpected keyword' in error_msg:
            log.warning(f'OpenAI client initialization issue: {error_msg}')
            log.info('Attempting initialization with custom HTTP client...')
            
            try:
                import httpx
                # Create httpx client without proxy configuration
                http_client = httpx.Client(
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    follow_redirects=True,
                    trust_env=False  # Disable automatic proxy detection from environment
                )
                openai_client = OpenAI(
                    api_key=openai_api_key,
                    http_client=http_client
                )
                
                restore_proxies()
                log.info('✓ OpenAI client initialized with custom HTTP client')
                openai_init_error = None
                return True
                
            except Exception as e2:
                error_msg2 = str(e2)
                openai_init_error = f"Primary error: {error_msg}, Fallback error: {error_msg2}"
                log.error(f'Failed with httpx fallback: {error_msg2}')
        else:
            openai_init_error = error_msg
            log.error(f'OpenAI initialization TypeError: {error_msg}')
        
        restore_proxies()
        log.warning('⚠ OpenAI client not available - will use fallback responses')
        return False
        
    except Exception as e:
        error_msg = str(e)
        openai_init_error = error_msg
        log.error(f'Failed to initialize OpenAI client: {error_msg}')
        log.warning('⚠ OpenAI client not available - will use fallback responses')
        restore_proxies()
        return False

# Initialize the client
initialize_openai_client()


def log_to_file(message):
    """Write log message to file"""
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")
    except Exception as e:
        log.error(f"Failed to write to log file: {e}")


def load_quizzes():
    """Load quiz configuration from JSON file"""
    try:
        with open(QUIZZES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        log.error(f"Quiz file not found: {QUIZZES_PATH}")
        return []
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in quiz file: {e}")
        return []


def load_prompts():
    """Load prompt configurations from JSON file"""
    try:
        with open(PROMPTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        log.error(f"Prompts file not found: {PROMPTS_PATH}")
        return []
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in prompts file: {e}")
        return []


def find_prompt_builder_by_category(category):
    """Find the promptBuilder for a given category from prompt.json"""
    if not category:
        return None
    
    prompts = load_prompts()
    if not prompts:
        return None
    
    # Normalize category for matching (lowercase, remove special chars)
    category_lower = category.lower().strip()
    
    # Category mapping for common variations
    category_mappings = {
        # AI related
        'ai-maturity': 'AI Adoption Readiness',
        'ai-adoption': 'AI Adoption Readiness',
        'ai adoption readiness': 'AI Adoption Readiness',
        'ai analysis': 'AI Adoption Readiness',
        'ai-analysis': 'AI Adoption Readiness',
        # Cyber Security
        'cyber': 'Cyber-Security Maturity',
        'cyber-security': 'Cyber-Security Maturity',
        'cybersecurity': 'Cyber-Security Maturity',
        'cyber-security maturity': 'Cyber-Security Maturity',
        # Cloud Adoption
        'cloud': 'Cloud Adoption Maturity',
        'cloud-adoption': 'Cloud Adoption Maturity',
        'cloud adoption maturity': 'Cloud Adoption Maturity',
        # Cloud Security
        'cloudsec': 'Cloud Security Maturity',
        'cloud-security': 'Cloud Security Maturity',
        'cloud security maturity': 'Cloud Security Maturity',
        # Hypervisor/Licensing
        'hyper': 'Hypervisor → Hyperscaler Licensing',
        'hypervisor': 'Hypervisor → Hyperscaler Licensing',
        'licensing': 'Hypervisor → Hyperscaler Licensing',
        # App Modernisation
        'modern': 'Application Modernisation',
        'app-modernisation': 'Application Modernisation',
        'application modernisation': 'Application Modernisation',
        # Managed Services
        'managed': 'Managed Services Readiness',
        'managed-services': 'Managed Services Readiness',
        'managed services readiness': 'Managed Services Readiness',
        # Data Security
        'datasec': 'Data Security Maturity',
        'data-security': 'Data Security Maturity',
        'data security maturity': 'Data Security Maturity',
    }
    
    # Try to find mapped category
    mapped_category = category_mappings.get(category_lower)
    
    for prompt in prompts:
        prompt_category = prompt.get('category', '')
        
        # Exact match (case-insensitive)
        if prompt_category.lower() == category_lower:
            log.info(f"Found exact promptBuilder match for category: {category}")
            return prompt.get('promptBuilder')
        
        # Mapped match
        if mapped_category and prompt_category == mapped_category:
            log.info(f"Found mapped promptBuilder for category: {category} -> {mapped_category}")
            return prompt.get('promptBuilder')
        
        # Partial match (category contains or is contained in prompt category)
        if category_lower in prompt_category.lower() or prompt_category.lower() in category_lower:
            log.info(f"Found partial promptBuilder match for category: {category} -> {prompt_category}")
            return prompt.get('promptBuilder')
    
    log.warning(f"No promptBuilder found for category: {category}")
    return None


# External Assessment API Configuration
EXTERNAL_API_BASE = os.getenv('EXTERNAL_API_BASE', 'https://assessments.botgo.io')


def is_valid_uuid(value):
    """Check if a string is a valid UUID"""
    import uuid as uuid_module
    try:
        uuid_module.UUID(str(value))
        return True
    except (ValueError, AttributeError):
        return False


def send_assessment_response_to_external_api(uuid, forms_data, response_data):
    """Send assessment response to external API"""
    try:
        url = f"{EXTERNAL_API_BASE}/api/assessment/response/{uuid}"
        log.info(f"Sending assessment response to external API: {url}")
        log_to_file(f"Sending assessment response to external API: {url}")
        
        # Build the payload
        payload = {
            "forms": forms_data,
            "response": response_data
        }
        
        # Convert payload to JSON bytes
        json_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        
        # Create request with proper headers
        req = urllib.request.Request(
            url, 
            data=json_data,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            log.info(f"Successfully sent assessment response for UUID: {uuid}")
            log_to_file(f"Successfully sent assessment response for UUID: {uuid}")
            log.info(f"External API response: {json.dumps(result)}")
            return result
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else ''
        log.error(f"HTTP error sending assessment response: {e.code} - {e.reason}")
        log.error(f"Error body: {error_body}")
        log_to_file(f"HTTP error sending assessment response: {e.code} - {e.reason}")
        return None
    except urllib.error.URLError as e:
        log.error(f"URL error sending assessment response: {e.reason}")
        log_to_file(f"URL error sending assessment response: {e.reason}")
        return None
    except Exception as e:
        log.error(f"Error sending assessment response to external API: {e}")
        log_to_file(f"Error sending assessment response to external API: {e}")
        return None


def fetch_assessment_from_external_api(uuid):
    """Fetch assessment info from external API by UUID"""
    try:
        url = f"{EXTERNAL_API_BASE}/api/assessment/info/{uuid}"
        log.info(f"Fetching assessment from external API: {url}")
        log_to_file(f"Fetching assessment from external API: {url}")
        
        import urllib.request
        import urllib.error
        
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            log.info(f"Successfully fetched assessment data for UUID: {uuid}")
            log_to_file(f"Successfully fetched assessment data for UUID: {uuid}")
            return data
    except urllib.error.HTTPError as e:
        log.error(f"HTTP error fetching assessment: {e.code} - {e.reason}")
        log_to_file(f"HTTP error fetching assessment: {e.code} - {e.reason}")
        return None
    except urllib.error.URLError as e:
        log.error(f"URL error fetching assessment: {e.reason}")
        log_to_file(f"URL error fetching assessment: {e.reason}")
        return None
    except json.JSONDecodeError as e:
        log.error(f"JSON decode error from external API: {e}")
        log_to_file(f"JSON decode error from external API: {e}")
        return None
    except Exception as e:
        log.error(f"Error fetching assessment from external API: {e}")
        log_to_file(f"Error fetching assessment from external API: {e}")
        return None


def transform_external_assessment_to_config(external_data, uuid):
    """Transform external API response to quiz config format"""
    if not external_data:
        return None
    
    # Handle nested response format: {"success": true, "data": {...}}
    if external_data.get('success') and external_data.get('data'):
        assessment_data = external_data['data']
    else:
        assessment_data = external_data
    
    # Extract questions from the external API response
    questions = assessment_data.get('questions', [])
    
    # Ensure questions is a list (not None)
    if not questions or not isinstance(questions, list):
        log.warning(f"No questions found in external API response for UUID: {uuid}")
        return None
    
    # Get category from external API for promptBuilder lookup
    category = assessment_data.get('category', '')
    log.info(f"External API category: {category}")
    log_to_file(f"External API category: {category}")
    
    # Look up promptBuilder by category from prompt.json
    prompt_builder = find_prompt_builder_by_category(category)
    
    # If no matching promptBuilder found, use default
    if not prompt_builder:
        log.info(f"Using default promptBuilder for category: {category}")
        prompt_builder = generate_default_prompt_builder(len(questions))
    else:
        log.info(f"Using category-specific promptBuilder for: {category}")
    
    # Convert emoji icons to font-awesome icons
    icon = assessment_data.get('icon', 'fa-clipboard-check')
    if not icon.startswith('fa-'):
        # Map common emojis to font-awesome icons
        emoji_to_fa = {
            '📝': 'fa-clipboard-check',
            '🧠': 'fa-brain',
            '☁️': 'fa-cloud',
            '🔒': 'fa-lock',
            '🛡️': 'fa-shield-halved',
            '💻': 'fa-laptop-code',
            '📊': 'fa-chart-bar',
            '🔧': 'fa-wrench',
            '⚙️': 'fa-cog',
            '🚀': 'fa-rocket',
        }
        icon = emoji_to_fa.get(icon, 'fa-clipboard-check')
    
    # Build a quiz config compatible with the frontend
    config = {
        'id': uuid,
        'title': assessment_data.get('title', 'Assessment'),
        'blurb': assessment_data.get('blurb', assessment_data.get('description', assessment_data.get('subtitle', 'Complete this assessment'))),
        'icon': icon,
        'color': assessment_data.get('color', '#0d6efd'),
        'questions': questions,
        'category': category,
        'promptBuilder': prompt_builder
    }
    
    log.info(f"Transformed external assessment: {config['title']} (category: {category}) with {len(questions)} questions")
    log_to_file(f"Transformed external assessment: {config['title']} (category: {category}) with {len(questions)} questions")
    
    return config


def generate_default_prompt_builder(num_questions):
    """Generate a default prompt builder function for assessments"""
    return f"""function(d){{ 
        const avg=(d.scores.reduce((a,b)=>a+b,0)/{num_questions}).toFixed(1); 
        const timelineMap={{'Immediate (0-3 months)':'3-month','Short-term (3-6 months)':'6-month','Medium-term (6-12 months)':'12-month','Long-term (12+ months)':'18-month','Exploratory phase':'6-month'}}; 
        const period=timelineMap[d.timeline]||'12-month'; 
        const quarters=period==='3-month'?{{phase1:'Month 1',phase2:'Month 2',phase3:'Month 3'}}:period==='6-month'?{{phase1:'Months 1-2',phase2:'Months 3-4',phase3:'Months 5-6'}}:period==='18-month'?{{q1:'Q1',q2:'Q2',q3:'Q3',q4:'Q4',q5:'Q5',q6:'Q6'}}:{{q1:'Q1',q2:'Q2',q3:'Q3',q4:'Q4'}}; 
        return 'You are a professional consultant. Create a comprehensive 900-word maturity assessment report for '+d.company+' ('+d.industry+' industry, '+d.employeeCount+' employees, '+d.annualRevenue+' revenue). Assessment score: '+avg+'/5. CRITICAL: Their selected implementation timeline is '+d.timeline+' with budget '+d.projectBudget+'. Provide 3 priority actions with realistic budget estimates. The contact is '+d.decisionMaker+'. IMPORTANT: Create a roadmap that matches their '+period+' timeline EXACTLY. Use these phases: '+JSON.stringify(quarters)+'. Return ONLY valid JSON with keys: executiveSummary, maturityLevel, overallScore, priorityActions (array with title, description, timeline, budget), roadmap (use the exact phase keys provided), recommendations (array), keyFindings (array), nextSteps (array).'; 
    }}"""


def call_openai(prompt):
    """Call OpenAI API to generate analysis in JSON format"""
    if not openai_client:
        raise ValueError("OpenAI client not initialized. Please set OPENAI_API_KEY in .env file")
    
    model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    log.info(f"OpenAI request - Model: {model}, Prompt length: {len(prompt)}")
    log_to_file(f"OpenAI request - Model: {model}, Prompt length: {len(prompt)}")
    
    # Enhance prompt to request JSON format
    json_prompt = f"""{prompt}

IMPORTANT: Respond ONLY with valid JSON in the following structure:
{{
  "executiveSummary": "Brief executive summary (2-3 sentences)",
  "maturityLevel": "Stage X: Name",
  "overallScore": X.X,
  "priorityActions": [
    {{
      "title": "Action title",
      "description": "Detailed description",
      "timeline": "Timeline (e.g., 'Q1 2024')",
      "budget": "Budget estimate if applicable"
    }}
  ],
  "roadmap": {{
    "q1": "Q1 activities and goals",
    "q2": "Q2 activities and goals",
    "q3": "Q3 activities and goals",
    "q4": "Q4 activities and goals"
  }},
  "recommendations": [
    "Recommendation 1",
    "Recommendation 2",
    "Recommendation 3"
  ],
  "keyFindings": [
    "Finding 1",
    "Finding 2",
    "Finding 3"
  ],
  "nextSteps": [
    "Step 1",
    "Step 2",
    "Step 3"
  ]
}}

Return ONLY the JSON object, no markdown formatting, no code blocks, no additional text."""
    
    try:
        # Build request parameters
        request_params = {
            "model": model,
            "messages": [
                {"role": "user", "content": json_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        # Add JSON response format for models that support it (gpt-3.5-turbo, gpt-4, etc.)
        # Check if model name contains supported prefixes
        if any(model.startswith(prefix) for prefix in ['gpt-3.5-turbo', 'gpt-4']):
            request_params["response_format"] = {"type": "json_object"}
        
        response = openai_client.chat.completions.create(**request_params)
        
        content = response.choices[0].message.content
        
        # Try to parse JSON
        try:
            # Remove markdown code blocks if present
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            parsed_json = json.loads(content)
            log.info(f"OpenAI response parsed successfully")
            log_to_file(f"OpenAI response parsed successfully")
            return parsed_json
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse JSON, returning raw content: {e}")
            log_to_file(f"Failed to parse JSON: {e}")
            # Fallback: return as structured error
            return {
                "error": "Failed to parse JSON response",
                "rawResponse": content,
                "executiveSummary": "Unable to parse AI response. Please try again.",
                "priorityActions": [],
                "roadmap": {},
                "recommendations": []
            }
            
    except Exception as e:
        log.error(f"OpenAI API error: {e}")
        log_to_file(f"OpenAI API error: {e}")
        raise


def parse_web_html_to_structured_data(html_content):
    """
    Parse HTML content from web display and convert to structured JSON format for PDF generation.
    
    Args:
        html_content (str): HTML content from the web display
        
    Returns:
        dict: Structured data matching the format expected by generate_pdf()
    """
    structured_data = {
        "executiveSummary": None,
        "maturityLevel": None,
        "overallScore": None,
        "priorityActions": [],
        "roadmap": {},
        "recommendations": [],
        "keyFindings": [],
        "nextSteps": []
    }
    
    if BS4_AVAILABLE:
        # Use BeautifulSoup for better HTML parsing
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract Executive Summary
        exec_summary_div = soup.find('div', style=re.compile(r'background.*#f0f7ff'))
        if exec_summary_div:
            exec_summary_p = exec_summary_div.find('p')
            if exec_summary_p:
                structured_data['executiveSummary'] = exec_summary_p.get_text(strip=True)
        
        # Extract Maturity Level and Overall Score
        flex_div = soup.find('div', style=re.compile(r'display.*flex'))
        if flex_div:
            maturity_div = flex_div.find('div', string=re.compile(r'Maturity Level'))
            if maturity_div:
                maturity_span = maturity_div.find_next('span', style=re.compile(r'color.*#0d6efd'))
                if maturity_span:
                    structured_data['maturityLevel'] = maturity_span.get_text(strip=True)
            
            score_div = flex_div.find('div', string=re.compile(r'Overall Score'))
            if score_div:
                score_span = score_div.find_next('span', style=re.compile(r'color.*#0d6efd'))
                if score_span:
                    score_text = score_span.get_text(strip=True)
                    # Extract numeric score (e.g., "3.5/5.0" -> 3.5)
                    score_match = re.search(r'(\d+\.?\d*)/', score_text)
                    if score_match:
                        structured_data['overallScore'] = float(score_match.group(1))
        
        # Extract Priority Actions
        priority_heading = soup.find('h3', string=re.compile(r'Priority Actions'))
        if priority_heading:
            actions_grid = priority_heading.find_next('div', style=re.compile(r'display.*grid'))
            if actions_grid:
                action_cards = actions_grid.find_all('div', style=re.compile(r'background.*#fff.*border'))
                for card in action_cards:
                    action = {}
                    # Extract title (h4 with blue color)
                    title_h4 = card.find('h4', style=re.compile(r'color.*#0d6efd'))
                    if title_h4:
                        title_text = title_h4.get_text(strip=True)
                        # Remove numbering (e.g., "1. " from "1. Action Title")
                        title_text = re.sub(r'^\d+\.\s*', '', title_text)
                        action['title'] = title_text
                    
                    # Extract description
                    desc_p = card.find('p', style=re.compile(r'margin.*0\.5rem'))
                    if desc_p:
                        action['description'] = desc_p.get_text(strip=True)
                    
                    # Extract timeline and budget
                    details_div = card.find('div', style=re.compile(r'display.*flex.*gap'))
                    if details_div:
                        timeline_span = details_div.find('span', string=re.compile(r'Timeline'))
                        if timeline_span:
                            timeline_text = timeline_span.get_text(strip=True)
                            timeline_match = re.search(r'Timeline:\s*(.+)', timeline_text)
                            if timeline_match:
                                action['timeline'] = timeline_match.group(1).strip()
                        
                        budget_span = details_div.find('span', string=re.compile(r'Budget'))
                        if budget_span:
                            budget_text = budget_span.get_text(strip=True)
                            budget_match = re.search(r'Budget:\s*(.+)', budget_text)
                            if budget_match:
                                action['budget'] = budget_match.group(1).strip()
                    
                    if action.get('title'):
                        structured_data['priorityActions'].append(action)
        
        # Extract Roadmap
        roadmap_heading = soup.find('h3', string=re.compile(r'12-Month Roadmap|Roadmap'))
        if roadmap_heading:
            roadmap_grid = roadmap_heading.find_next('div', style=re.compile(r'display.*grid'))
            if roadmap_grid:
                roadmap_cards = roadmap_grid.find_all('div', style=re.compile(r'background.*#f8f9fa'))
                for card in roadmap_cards:
                    quarter_strong = card.find('strong', style=re.compile(r'color.*#0d6efd'))
                    if quarter_strong:
                        quarter = quarter_strong.get_text(strip=True).lower()
                        content_p = card.find('p')
                        if content_p:
                            structured_data['roadmap'][quarter] = content_p.get_text(strip=True)
        
        # Extract Recommendations
        rec_heading = soup.find('h3', string=re.compile(r'Recommendations'))
        if rec_heading:
            rec_ul = rec_heading.find_next('ul')
            if rec_ul:
                rec_items = rec_ul.find_all('li')
                structured_data['recommendations'] = [li.get_text(strip=True) for li in rec_items]
        
        # Extract Key Findings
        findings_heading = soup.find('h3', string=re.compile(r'Key Findings'))
        if findings_heading:
            findings_ul = findings_heading.find_next('ul')
            if findings_ul:
                findings_items = findings_ul.find_all('li')
                structured_data['keyFindings'] = [li.get_text(strip=True) for li in findings_items]
        
        # Extract Next Steps
        steps_heading = soup.find('h3', string=re.compile(r'Next Steps'))
        if steps_heading:
            steps_ul = steps_heading.find_next('ul')
            if steps_ul:
                steps_items = steps_ul.find_all('li')
                structured_data['nextSteps'] = [li.get_text(strip=True) for li in steps_items]
    
    else:
        # Fallback: Use regex parsing if BeautifulSoup is not available
        log.warning('Using regex fallback for HTML parsing. Install beautifulsoup4 for better results.')
        
        # Extract Executive Summary
        exec_match = re.search(r'Executive Summary[^<]*</h3>\s*<p>([^<]+)</p>', html_content, re.IGNORECASE)
        if exec_match:
            structured_data['executiveSummary'] = exec_match.group(1).strip()
        
        # Extract Maturity Level
        maturity_match = re.search(r'Maturity Level:[^<]*<span[^>]*>([^<]+)</span>', html_content, re.IGNORECASE)
        if maturity_match:
            structured_data['maturityLevel'] = maturity_match.group(1).strip()
        
        # Extract Overall Score
        score_match = re.search(r'Overall Score:[^<]*<span[^>]*>([^<]+)/5\.0</span>', html_content, re.IGNORECASE)
        if score_match:
            score_text = score_match.group(1).strip()
            try:
                structured_data['overallScore'] = float(score_text)
            except ValueError:
                pass
        
        # Extract Priority Actions (simplified regex)
        action_pattern = r'<h4[^>]*>(\d+)\.\s*([^<]+)</h4>'
        actions = re.finditer(action_pattern, html_content)
        for action_match in actions:
            action = {
                'title': action_match.group(2).strip()
            }
            # Try to extract description (next <p> tag)
            desc_match = re.search(r'</h4>\s*<p[^>]*>([^<]+)</p>', html_content[action_match.end():])
            if desc_match:
                action['description'] = desc_match.group(1).strip()
            structured_data['priorityActions'].append(action)
        
        # Extract Recommendations
        rec_pattern = r'<li[^>]*>([^<]+)</li>'
        rec_section = re.search(r'Recommendations[^<]*</h3>\s*<ul[^>]*>(.*?)</ul>', html_content, re.DOTALL | re.IGNORECASE)
        if rec_section:
            rec_items = re.findall(rec_pattern, rec_section.group(2))
            structured_data['recommendations'] = [item.strip() for item in rec_items]
        
        # Extract Key Findings
        findings_section = re.search(r'Key Findings[^<]*</h3>\s*<ul[^>]*>(.*?)</ul>', html_content, re.DOTALL | re.IGNORECASE)
        if findings_section:
            findings_items = re.findall(rec_pattern, findings_section.group(2))
            structured_data['keyFindings'] = [item.strip() for item in findings_items]
        
        # Extract Next Steps
        steps_section = re.search(r'Next Steps[^<]*</h3>\s*<ul[^>]*>(.*?)</ul>', html_content, re.DOTALL | re.IGNORECASE)
        if steps_section:
            steps_items = re.findall(rec_pattern, steps_section.group(2))
            structured_data['nextSteps'] = [item.strip() for item in steps_items]
    
    # Clean up empty values
    structured_data = {k: v for k, v in structured_data.items() if v}
    
    log.info('Parsed HTML content to structured data')
    log_to_file('Parsed HTML content to structured data')
    
    return structured_data


def create_category_chart(user_scores, industry_scores):
    """Create a bar chart comparing user scores vs industry scores"""
    try:
        categories = list(user_scores.keys())
        user_values = [user_scores.get(cat, 0) for cat in categories]
        industry_values = [industry_scores.get(cat, 0) for cat in categories]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(categories))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], user_values, width, label='Your Score', color='#0d6efd', alpha=0.85, edgecolor='white', linewidth=1.5)
        bars2 = ax.bar([i + width/2 for i in x], industry_values, width, label='Industry Average', color='#28a745', alpha=0.85, edgecolor='white', linewidth=1.5)
        
        # Add value labels on bars
        for i, (u_val, i_val) in enumerate(zip(user_values, industry_values)):
            ax.text(i - width/2, u_val + 0.1, f'{u_val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.text(i + width/2, i_val + 0.1, f'{i_val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Category', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title('Category Comparison', fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=15, ha='right', fontsize=9)
        ax.set_ylim(0, 5.5)
        ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    except Exception as e:
        log.error(f'Error creating category chart: {e}')
        return None


def create_trend_chart(trend_data):
    """Create a line chart showing industry trends"""
    try:
        if not trend_data or len(trend_data) == 0:
            return None
        
        scores = [item.get('score', 0) for item in trend_data]
        
        # Convert month numbers to names with year handling
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_labels = []
        current_year = datetime.now().year
        
        for item in trend_data:
            month = item.get('month', 1)
            year = item.get('year', current_year)
            month_name = month_names[month-1] if 1 <= month <= 12 else f'M{month}'
            
            # Show year only if it's different from current year
            if year != current_year:
                month_labels.append(f"{month_name} '{str(year)[-2:]}")
            else:
                month_labels.append(month_name)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(month_labels, scores, marker='o', linewidth=2.5, markersize=8, color='#28a745', label='Industry Trend', markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(month_labels, scores, alpha=0.2, color='#28a745')
        
        ax.set_xlabel('Month', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_title('Industry Trend (Last 7 Months)', fontsize=12, fontweight='bold', pad=15)
        ax.set_ylim(0, 5.5)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    except Exception as e:
        log.error(f'Error creating trend chart: {e}')
        return None


def create_maturity_gauge(score):
    """Create a visual maturity level gauge/progress bar"""
    try:
        fig, ax = plt.subplots(figsize=(6, 2))
        
        # Calculate percentage
        percentage = (score / 5.0) * 100
        
        # Color based on score
        if score < 2:
            bar_color = '#dc3545'  # Red
        elif score < 3:
            bar_color = '#ffc107'  # Yellow
        elif score < 4:
            bar_color = '#17a2b8'  # Cyan
        else:
            bar_color = '#28a745'  # Green
        
        # Create horizontal bar
        ax.barh(0, percentage, height=0.6, color=bar_color, alpha=0.8, edgecolor='white', linewidth=2)
        ax.barh(0, 100, height=0.6, color='#e9ecef', alpha=0.3, edgecolor='#dee2e6')
        
        # Add text
        ax.text(percentage/2, 0, f'{score}/5.0', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax.text(50, -0.8, 'Maturity Level', ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    except Exception as e:
        log.error(f'Error creating maturity gauge: {e}')
        return None


def create_priority_matrix(actions):
    """Create a priority matrix visualization"""
    try:
        if not actions or len(actions) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Extract priorities (simplified - using index as priority indicator)
        # In real scenario, you'd have impact/effort scores
        x_pos = []
        y_pos = []
        labels = []
        
        for i, action in enumerate(actions[:6]):  # Limit to 6 for readability
            # Simulate impact and effort (in real scenario, these would come from data)
            impact = 3 + random.uniform(-1, 1)
            effort = 2 + random.uniform(-0.5, 1.5)
            x_pos.append(impact)
            y_pos.append(effort)
            labels.append(action.get('title', f'Action {i+1}')[:20])
        
        # Create scatter plot
        scatter = ax.scatter(x_pos, y_pos, s=200, alpha=0.6, c=range(len(x_pos)), cmap='viridis', edgecolors='black', linewidth=1.5)
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (x_pos[i], y_pos[i]), xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Impact', fontsize=11, fontweight='bold')
        ax.set_ylabel('Effort', fontsize=11, fontweight='bold')
        ax.set_title('Action Priority Matrix', fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(1, 5)
        ax.set_ylim(1, 5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    except Exception as e:
        log.error(f'Error creating priority matrix: {e}')
        return None


def generate_pdf_from_web_html(html_content, user_info, quiz_title):
    """
    Generate PDF from web HTML content by parsing it to structured data first.
    
    Args:
        html_content (str): HTML content from web display
        user_info (dict): User information (name, email, company, industry)
        quiz_title (str): Title of the quiz/assessment
        
    Returns:
        BytesIO: PDF buffer
    """
    # Parse HTML to structured data
    structured_data = parse_web_html_to_structured_data(html_content)
    
    # Generate PDF using the structured data (no benchmark data available from HTML parsing)
    return generate_pdf(structured_data, user_info, quiz_title, None)


def generate_pdf(report_data, user_info, quiz_title, benchmark_data=None):
    """Generate attractive, professional PDF report with enhanced visuals and dual logos"""
    buffer = BytesIO()
    # Clean margins for professional look
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.7*inch, leftMargin=0.8*inch, rightMargin=0.8*inch)
    
    # Track temp files for cleanup after PDF generation
    temp_files_to_cleanup = []
    
    # Helper function to get roadmap title based on timeline
    def get_roadmap_title():
        timeline = user_info.get('timeline', '')
        timeline_map = {
            'Immediate (0-3 months)': '3-MONTH',
            'Short-term (3-6 months)': '6-MONTH',
            'Medium-term (6-12 months)': '12-MONTH',
            'Long-term (12+ months)': '18-MONTH',
            'Exploratory phase': '6-MONTH'
        }
        return timeline_map.get(timeline, '12-MONTH')
    
    story = []
    styles = getSampleStyleSheet()
    
    # Enhanced professional styles with better typography
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=26,
        textColor=colors.HexColor('#000000'),
        spaceAfter=15,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold',
        leading=30
    )
    
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#000000'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold',
        leading=20,
        borderWidth=0,
        borderPadding=0
    )
    
    subheading_style = ParagraphStyle(
        'SubHeadingStyle',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=colors.HexColor('#495057'),
        spaceAfter=8,
        spaceBefore=14,
        fontName='Helvetica-Bold',
        leading=16
    )
    
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor('#212529'),
        spaceAfter=8
    )
    
    # ====================
    # HEADER WITH DUAL LOGOS
    # ====================
    header_content = []
    
    # Try to load logos from S3
    logo1_url = 'https://botgobucket.s3.ap-south-1.amazonaws.com/maturity-assessment/glb-logo.png'
    logo2_url = 'https://botgobucket.s3.ap-south-1.amazonaws.com/maturity-assessment/logo-kergan-black.png'
    
    # Helper function to download image from URL
    def download_image(url):
        try:
            with urllib.request.urlopen(url) as response:
                img_data = response.read()
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_file.write(img_data)
                temp_file.close()
                return temp_file.name
        except Exception as e:
            log.warning(f'Could not download image from {url}: {e}')
            return None
    
    logos_row = []
    
    # Load Globtier logo (smaller size)
    logo1_temp = download_image(logo1_url)
    if logo1_temp:
        try:
            logo1 = Image(logo1_temp, width=1.0*inch, height=0.5*inch)
            logos_row.append(logo1)
            # Track temp file for cleanup after PDF generation
            temp_files_to_cleanup.append(logo1_temp)
        except Exception as e:
            log.warning(f'Could not load Globtier logo: {e}')
            logos_row.append(Paragraph("<b>GLOBTIER</b>", ParagraphStyle('Logo1', parent=normal_style, fontSize=16, fontName='Helvetica-Bold', textColor=colors.HexColor('#000000'))))
            if logo1_temp:
                try:
                    os.unlink(logo1_temp)
                except:
                    pass
    else:
        # Fallback to local file if S3 download fails
        logo1_path = 'logo/glb-logo.png'
        if os.path.exists(logo1_path):
            try:
                logo1 = Image(logo1_path, width=1.0*inch, height=0.5*inch)
                logos_row.append(logo1)
            except Exception as e:
                log.warning(f'Could not load Globtier logo from local: {e}')
                logos_row.append(Paragraph("<b>GLOBTIER</b>", ParagraphStyle('Logo1', parent=normal_style, fontSize=16, fontName='Helvetica-Bold', textColor=colors.HexColor('#000000'))))
        else:
            logos_row.append(Paragraph("<b>GLOBTIER</b>", ParagraphStyle('Logo1', parent=normal_style, fontSize=16, fontName='Helvetica-Bold', textColor=colors.HexColor('#000000'))))
    
    # Center text
    logos_row.append(Paragraph("<font size='18' color='black'><b>Maturity Assessment Report</b></font>", 
                               ParagraphStyle('CenterTitle', parent=normal_style, alignment=TA_CENTER, fontSize=12)))
    
    # Load Botgo logo (PNG from S3) (smaller size)
    logo2_temp = download_image(logo2_url)
    if logo2_temp:
        try:
            logo2 = Image(logo2_temp, width=1.0*inch, height=0.5*inch)
            logos_row.append(logo2)
            # Track temp file for cleanup after PDF generation
            temp_files_to_cleanup.append(logo2_temp)
        except Exception as e:
            log.warning(f'Could not load Botgo logo: {e}')
            logos_row.append(Paragraph("<b>BOTGO</b>", ParagraphStyle('Logo2', parent=normal_style, fontSize=16, fontName='Helvetica-Bold', textColor=colors.HexColor('#000000'), alignment=TA_RIGHT)))
            if logo2_temp:
                try:
                    os.unlink(logo2_temp)
                except:
                    pass
    else:
        # Fallback to local file if S3 download fails
        logo2_path = 'logo/logo-kergan-black.png'
        if os.path.exists(logo2_path):
            try:
                logo2 = Image(logo2_path, width=1.0*inch, height=0.5*inch)
                logos_row.append(logo2)
            except Exception as e:
                log.warning(f'Could not load Botgo logo from local: {e}')
                logos_row.append(Paragraph("<b>BOTGO</b>", ParagraphStyle('Logo2', parent=normal_style, fontSize=16, fontName='Helvetica-Bold', textColor=colors.HexColor('#000000'), alignment=TA_RIGHT)))
        else:
            log.warning(f'Botgo logo not found at: {logo2_path}')
            logos_row.append(Paragraph("<b>BOTGO</b>", ParagraphStyle('Logo2', parent=normal_style, fontSize=16, fontName='Helvetica-Bold', textColor=colors.HexColor('#000000'), alignment=TA_RIGHT)))
    
    header_table = Table([logos_row], colWidths=[1.8*inch, 2.8*inch, 1.8*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ('ALIGN', (2, 0), (2, 0), 'RIGHT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.15*inch))
    
    # ====================
    # ASSESSMENT TITLE BAR
    # ====================
    title_data = [
        [Paragraph(f"<font size='22' color='white'><b>{quiz_title}</b></font><br/><font size='10' color='white'>Comprehensive Assessment Report</font>", 
                  ParagraphStyle('TitleHeader', parent=normal_style, alignment=TA_CENTER, fontSize=12, textColor=colors.white))]
    ]
    title_table = Table(title_data, colWidths=[6.4*inch])
    title_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#000000')),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('TOPPADDING', (0, 0), (-1, -1), 20),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(title_table)
    story.append(Spacer(1, 0.25*inch))
    
    # ====================
    # SECTION 1: COMPANY INFORMATION
    # ====================
    story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                          ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("SECTION 1: ORGANIZATION PROFILE", 
                          ParagraphStyle('SectionHeader', parent=heading_style, fontSize=14, textColor=colors.HexColor('#000000'), 
                                       fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=10)))
    story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                          ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
    story.append(Spacer(1, 0.15*inch))
    
    # Enhanced Company info box with better formatting
    company_data = [
        [Paragraph("<b>Organization Name</b>", ParagraphStyle('InfoLabel', parent=normal_style, fontSize=9, textColor=colors.HexColor('#666'))),
         Paragraph(f"{user_info.get('company', 'N/A')}", ParagraphStyle('InfoValue', parent=normal_style, fontSize=10, fontName='Helvetica-Bold'))],
        [Paragraph("<b>Industry Sector</b>", ParagraphStyle('InfoLabel', parent=normal_style, fontSize=9, textColor=colors.HexColor('#666'))),
         Paragraph(f"{user_info.get('industry', 'N/A')}", ParagraphStyle('InfoValue', parent=normal_style, fontSize=10))],
        [Paragraph("<b>Report Generated</b>", ParagraphStyle('InfoLabel', parent=normal_style, fontSize=9, textColor=colors.HexColor('#666'))),
         Paragraph(f"{datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ParagraphStyle('InfoValue', parent=normal_style, fontSize=10))],
        [Paragraph("<b>Contact Person</b>", ParagraphStyle('InfoLabel', parent=normal_style, fontSize=9, textColor=colors.HexColor('#666'))),
         Paragraph(f"{user_info.get('name', 'N/A')}", ParagraphStyle('InfoValue', parent=normal_style, fontSize=10))]
    ]
    company_table = Table(company_data, colWidths=[2.0*inch, 4.4*inch])
    company_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
    ]))
    story.append(company_table)
    story.append(Spacer(1, 0.3*inch))
    
    # ====================
    # SECTION 2: EXECUTIVE SUMMARY
    # ====================
    if report_data.get('executiveSummary'):
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("SECTION 2: EXECUTIVE SUMMARY", 
                              ParagraphStyle('SectionHeader', parent=heading_style, fontSize=14, textColor=colors.HexColor('#000000'), 
                                           fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=10)))
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.15*inch))
        
        exec_summary_data = [
            [Paragraph(report_data['executiveSummary'], ParagraphStyle('ExecText', parent=normal_style, spaceAfter=0, leading=16, alignment=TA_JUSTIFY))]
        ]
        exec_table = Table(exec_summary_data, colWidths=[6.4*inch])
        exec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fffaf0')),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBEFORE', (0, 0), (0, -1), 4, colors.HexColor('#000000')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
        ]))
        story.append(exec_table)
        story.append(Spacer(1, 0.3*inch))
    
    # ====================
    # SECTION 3: ASSESSMENT OVERVIEW
    # ====================
    if report_data.get('maturityLevel') or report_data.get('overallScore') is not None:
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("SECTION 3: ASSESSMENT OVERVIEW", 
                              ParagraphStyle('SectionHeader', parent=heading_style, fontSize=14, textColor=colors.HexColor('#000000'), 
                                           fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=10)))
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.15*inch))
        
        # Create maturity gauge if score available
        overall_score = report_data.get('overallScore', 0)
        if overall_score is not None:
            gauge_buffer = create_maturity_gauge(overall_score)
            if gauge_buffer:
                try:
                    gauge_img = Image(gauge_buffer, width=6.4*inch, height=1.5*inch)
                    story.append(gauge_img)
                    story.append(Spacer(1, 0.15*inch))
                except Exception as e:
                    log.warning(f'Could not add maturity gauge: {e}')
        
        # Maturity level and score info boxes
        maturity_info_data = []
        if report_data.get('maturityLevel'):
            maturity_info_data.append([
                Paragraph(f"<b>Maturity Level</b><br/><font size='14' color='#0d6efd'>{report_data['maturityLevel']}</font>", 
                         ParagraphStyle('InfoBox', parent=normal_style, alignment=TA_CENTER, fontSize=10))
            ])
        if report_data.get('overallScore') is not None:
            maturity_info_data.append([
                Paragraph(f"<b>Overall Score</b><br/><font size='14' color='#0d6efd'>{overall_score}/5.0</font>", 
                         ParagraphStyle('InfoBox', parent=normal_style, alignment=TA_CENTER, fontSize=10))
            ])
        
        if maturity_info_data:
            maturity_table = Table(maturity_info_data, colWidths=[3.2*inch] * len(maturity_info_data))
            maturity_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 15),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ]))
            story.append(maturity_table)
            story.append(Spacer(1, 0.3*inch))
    
    # Benchmark Comparison Section (if available) - Enhanced format matching web display
    if benchmark_data:
        story.append(Paragraph(f"Performance vs. {benchmark_data.get('industry', 'Industry')} Industry", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Score boxes - matching web format
        user_score = benchmark_data.get('userScore', 0)
        industry_avg = benchmark_data.get('industryAverage', 0)
        percentile = benchmark_data.get('percentile', 'N/A')
        industry_name = benchmark_data.get('industry', 'Industry')
        
        # Create score boxes table
        score_boxes_data = [
            [
                Paragraph(
                    "<b>YOUR SCORE</b><br/><br/><font size='20' color='#0d6efd'><b>{}/5.0</b></font><br/><br/><font size='9' color='#666666'>{}{} Percentile</font>".format(
                        user_score, 
                        percentile,
                        'th' if isinstance(percentile, (int, float)) and percentile != 'N/A' else ''
                    ),
                    ParagraphStyle('ScoreBox', parent=normal_style, alignment=TA_CENTER, fontSize=10, spaceAfter=0)
                ),
                Paragraph(
                    "<b>INDUSTRY AVERAGE</b><br/><br/><font size='20' color='#28a745'><b>{}/5.0</b></font><br/><br/><font size='9' color='#666666'>{}{} Sector</font>".format(
                        industry_avg,
                        industry_name,
                        ' ' if industry_name else ''
                    ),
                    ParagraphStyle('ScoreBox', parent=normal_style, alignment=TA_CENTER, fontSize=10, spaceAfter=0)
                )
            ]
        ]
        
        score_boxes_table = Table(score_boxes_data, colWidths=[3.2*inch, 3.2*inch])
        score_boxes_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#e7f3ff')),
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#d4edda')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 20),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ]))
        story.append(score_boxes_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Category comparison chart
        if benchmark_data.get('userCategoryScores') and benchmark_data.get('industryCategoryScores'):
            user_cats = benchmark_data.get('userCategoryScores', {})
            industry_cats = benchmark_data.get('industryCategoryScores', {})
            
            # Create chart
            chart_buffer = create_category_chart(user_cats, industry_cats)
            if chart_buffer:
                try:
                    chart_img = Image(chart_buffer, width=6.4*inch, height=4*inch)
                    story.append(Paragraph("<b>Category Comparison</b>", ParagraphStyle('ChartTitle', parent=normal_style, fontSize=11, spaceAfter=8, spaceBefore=12)))
                    story.append(chart_img)
                    story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    log.warning(f'Could not add category chart to PDF: {e}')
            
            # Also include table for reference
            category_data = [['Category', 'Your Score', f'{industry_name} Avg']]
            for category in user_cats.keys():
                user_score_cat = user_cats.get(category, 0)
                industry_score_cat = industry_cats.get(category, 0)
                category_data.append([category, f'{user_score_cat}/5.0', f'{industry_score_cat}/5.0'])
            
            category_table = Table(category_data, colWidths=[2.8*inch, 1.8*inch, 1.8*inch])
            category_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d6efd')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ]))
            story.append(category_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Industry trend chart
        if benchmark_data.get('trendData') and len(benchmark_data.get('trendData', [])) > 0:
            trend_buffer = create_trend_chart(benchmark_data.get('trendData'))
            if trend_buffer:
                try:
                    trend_img = Image(trend_buffer, width=6.4*inch, height=3.2*inch)
                    story.append(Paragraph("<b>Industry Trend (Last 7 Months)</b>", ParagraphStyle('ChartTitle', parent=normal_style, fontSize=11, spaceAfter=8, spaceBefore=12)))
                    story.append(trend_img)
                    story.append(Spacer(1, 0.3*inch))
                except Exception as e:
                    log.warning(f'Could not add trend chart to PDF: {e}')
    
    # ====================
    # SECTION 4: PRIORITY ACTIONS
    # ====================
    if report_data.get('priorityActions') and len(report_data['priorityActions']) > 0:
        story.append(PageBreak())
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("SECTION 4: PRIORITY ACTIONS", 
                              ParagraphStyle('SectionHeader', parent=heading_style, fontSize=14, textColor=colors.HexColor('#000000'), 
                                           fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=10)))
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.15*inch))
        
        # Add priority matrix visualization
        priority_matrix_buffer = create_priority_matrix(report_data['priorityActions'])
        if priority_matrix_buffer:
            try:
                matrix_img = Image(priority_matrix_buffer, width=5*inch, height=4*inch)
                story.append(Paragraph("<b>Action Priority Matrix</b>", ParagraphStyle('ChartTitle', parent=normal_style, fontSize=11, spaceAfter=8, spaceBefore=12)))
                story.append(matrix_img)
                story.append(Spacer(1, 0.3*inch))
            except Exception as e:
                log.warning(f'Could not add priority matrix: {e}')
        
        # Enhanced action cards with better styling
        for idx, action in enumerate(report_data['priorityActions'], 1):
            action_title = f"{idx}. {action.get('title', 'Action ' + str(idx))}"
            action_desc = action.get('description', '')
            action_details = []
            if action.get('timeline'):
                action_details.append(f"⏱ <b>Timeline:</b> {action['timeline']}")
            if action.get('budget'):
                action_details.append(f"💰 <b>Budget:</b> {action['budget']}")
            
            # Create action card
            action_card_data = [[
                Paragraph(
                    f"<font size='12' color='#0d6efd'><b>{action_title}</b></font><br/><br/>{action_desc}<br/><br/><font size='9' color='#666666'>{' &nbsp;&nbsp; '.join(action_details)}</font>",
                    ParagraphStyle('ActionCard', parent=normal_style, fontSize=10.5, spaceAfter=0, leading=16)
                )
            ]]
            
            action_card_table = Table(action_card_data, colWidths=[6.4*inch])
            action_card_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                ('TOPPADDING', (0, 0), (-1, -1), 15),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('LINEBELOW', (0, 0), (0, 0), 3, colors.HexColor('#0d6efd')),
            ]))
            story.append(action_card_table)
            if idx < len(report_data['priorityActions']):
                story.append(Spacer(1, 0.15*inch))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Add ROI Projections section
        story.append(Paragraph("💡 Expected Impact & ROI", subheading_style))
        roi_text = """
        Based on industry benchmarks and similar implementations, the priority actions outlined above are projected to deliver:
        <br/><br/>
        • <b>Short-term (3-6 months):</b> Improved operational efficiency and foundational capabilities
        <br/>
        • <b>Medium-term (6-12 months):</b> Measurable improvements in key performance indicators
        <br/>
        • <b>Long-term (12+ months):</b> Competitive advantages and sustainable growth
        """
        roi_data = [[Paragraph(roi_text, ParagraphStyle('ROIText', parent=normal_style, fontSize=10, spaceAfter=0, leading=16))]]
        roi_table = Table(roi_data, colWidths=[6.4*inch])
        roi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff3cd')),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ffc107')),
        ]))
        story.append(roi_table)
        story.append(Spacer(1, 0.2*inch))
    
    # ====================
    # SECTION 5: STRATEGIC ROADMAP (Dynamic based on timeline)
    # ====================
    if report_data.get('roadmap') and len(report_data['roadmap']) > 0:
        roadmap_title = get_roadmap_title()
        story.append(PageBreak())
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"SECTION 5: {roadmap_title} STRATEGIC ROADMAP", 
                              ParagraphStyle('SectionHeader', parent=heading_style, fontSize=14, textColor=colors.HexColor('#000000'), 
                                           fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=10)))
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.15*inch))
        
        # Enhanced roadmap with colored quarters
        roadmap_data = []
        quarter_colors = {
            'q1': colors.HexColor('#e7f3ff'),
            'q2': colors.HexColor('#d4edda'),
            'q3': colors.HexColor('#fff3cd'),
            'q4': colors.HexColor('#f8d7da')
        }
        
        for idx, (quarter, content) in enumerate(report_data['roadmap'].items()):
            bg_color = quarter_colors.get(quarter.lower(), colors.HexColor('#f8f9fa'))
            roadmap_data.append([
                Paragraph(
                    f"<b><font size='13' color='#0d6efd'>{quarter.upper()}</font></b>",
                    ParagraphStyle('RoadmapQ', parent=normal_style, fontSize=12, textColor=colors.HexColor('#0d6efd'), spaceAfter=0, spaceBefore=0, alignment=TA_CENTER)
                ),
                Paragraph(
                    content,
                    ParagraphStyle('RoadmapText', parent=normal_style, fontSize=10.5, spaceAfter=0, leading=16)
                )
            ])
        
        roadmap_table = Table(roadmap_data, colWidths=[1.3*inch, 5.1*inch])
        roadmap_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#0d6efd')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#f8f9fa')),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ]))
        story.append(roadmap_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Add Risk & Opportunity Analysis
        story.append(Paragraph("⚠️ Risk & Opportunity Analysis", subheading_style))
        risk_text = """
        <b>Key Risks to Address:</b>
        <br/>• Delayed implementation may result in competitive disadvantage
        <br/>• Resource constraints could impact timeline and quality
        <br/>• Change management challenges may affect adoption rates
        <br/><br/>
        <b>Opportunities to Leverage:</b>
        <br/>• Early adoption can establish market leadership position
        <br/>• Strategic partnerships can accelerate implementation
        <br/>• Incremental approach allows for learning and optimization
        """
        risk_data = [[Paragraph(risk_text, ParagraphStyle('RiskText', parent=normal_style, fontSize=10, spaceAfter=0, leading=16))]]
        risk_table = Table(risk_data, colWidths=[6.4*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8d7da')),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dc3545')),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 0.2*inch))
    
    # ====================
    # SECTION 6: RECOMMENDATIONS
    # ====================
    if report_data.get('recommendations') and len(report_data['recommendations']) > 0:
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("SECTION 6: RECOMMENDATIONS", 
                              ParagraphStyle('SectionHeader', parent=heading_style, fontSize=14, textColor=colors.HexColor('#000000'), 
                                           fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=10)))
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.15*inch))
        rec_data = []
        for rec in report_data['recommendations']:
            rec_data.append([
                Paragraph(f"✓ {rec}", ParagraphStyle('RecItem', parent=normal_style, fontSize=10.5, spaceAfter=8, spaceBefore=0, leading=16))
            ])
        
        rec_table = Table(rec_data, colWidths=[6.4*inch])
        rec_table.setStyle(TableStyle([
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 0.2*inch))
    
    # ====================
    # SECTION 7: KEY FINDINGS
    # ====================
    if report_data.get('keyFindings') and len(report_data['keyFindings']) > 0:
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("SECTION 7: KEY FINDINGS", 
                              ParagraphStyle('SectionHeader', parent=heading_style, fontSize=14, textColor=colors.HexColor('#000000'), 
                                           fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=10)))
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.15*inch))
        findings_data = []
        for finding in report_data['keyFindings']:
            findings_data.append([
                Paragraph(f"• {finding}", ParagraphStyle('FindingItem', parent=normal_style, fontSize=10.5, spaceAfter=8, spaceBefore=0, leading=16))
            ])
        
        findings_table = Table(findings_data, colWidths=[6.4*inch])
        findings_table.setStyle(TableStyle([
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff3cd')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ffc107')),
        ]))
        story.append(findings_table)
        story.append(Spacer(1, 0.2*inch))
    
    # ====================
    # SECTION 8: NEXT STEPS
    # ====================
    if report_data.get('nextSteps') and len(report_data['nextSteps']) > 0:
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("SECTION 8: NEXT STEPS", 
                              ParagraphStyle('SectionHeader', parent=heading_style, fontSize=14, textColor=colors.HexColor('#000000'), 
                                           fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=10)))
        story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                              ParagraphStyle('Divider', parent=normal_style, textColor=colors.HexColor('#dee2e6'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.15*inch))
        steps_data = []
        for step in report_data['nextSteps']:
            steps_data.append([
                Paragraph(f"→ {step}", ParagraphStyle('StepItem', parent=normal_style, fontSize=10.5, spaceAfter=8, spaceBefore=0, leading=16))
            ])
        
        steps_table = Table(steps_data, colWidths=[6.4*inch])
        steps_table.setStyle(TableStyle([
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#d1ecf1')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#17a2b8')),
        ]))
        story.append(steps_table)
        story.append(Spacer(1, 0.2*inch))
    
    # ====================
    # SECTION 9: ABOUT GLOBTIER & BOTGO (with black background and white text)
    # ====================
    story.append(PageBreak())
    
    # Create white text styles for dark background
    white_heading_style = ParagraphStyle(
        'WhiteHeading',
        parent=heading_style,
        fontSize=14,
        textColor=colors.white,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER,
        spaceAfter=10
    )
    
    white_subhead_style = ParagraphStyle(
        'WhiteSubhead',
        parent=subheading_style,
        fontSize=14,
        textColor=colors.white,
        fontName='Helvetica-Bold',
        spaceAfter=8
    )
    
    white_normal_style = ParagraphStyle(
        'WhiteNormal',
        parent=normal_style,
        textColor=colors.white,
        fontSize=10.5,
        leading=16
    )
    
    # Header with white divider
    story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                          ParagraphStyle('WhiteDivider', parent=normal_style, textColor=colors.white, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("ABOUT GLOBTIER & BOTGO", white_heading_style))
    story.append(Paragraph("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
                          ParagraphStyle('WhiteDivider', parent=normal_style, textColor=colors.white, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.15*inch))
    
    # Create optimized paragraph styles with controlled spacing
    white_subhead_tight = ParagraphStyle(
        'WhiteSubheadTight',
        parent=white_subhead_style,
        spaceAfter=6,
        spaceBefore=0
    )
    
    white_normal_tight = ParagraphStyle(
        'WhiteNormalTight',
        parent=white_normal_style,
        spaceAfter=10,
        spaceBefore=0
    )
    
    # Create single comprehensive table with all content and black background
    about_content = []
    
    # Globtier section
    about_content.append([Paragraph("<b>Globtier</b>", white_subhead_tight)])
    globtier_text = "Globtier is a leading technology consulting firm specializing in digital transformation, cloud adoption, cybersecurity, and AI implementation. With decades of combined experience, our team of certified experts helps organizations navigate complex technology challenges and achieve measurable business outcomes."
    about_content.append([Paragraph(globtier_text, white_normal_tight)])
    
    # Botgo section
    about_content.append([Paragraph("<b>Botgo</b>", white_subhead_tight)])
    botgo_text = "Botgo is our intelligent automation and AI platform that powers assessment tools, business process automation, and intelligent decision support systems. Botgo leverages cutting-edge AI technologies to deliver personalized insights and recommendations at scale."
    about_content.append([Paragraph(botgo_text, white_normal_tight)])
    
    # Our Services section
    about_content.append([Paragraph("<b>Our Services</b>", white_subhead_tight)])
    services = [
        "• <b>Cloud Migration & Optimization:</b> End-to-end cloud strategy, migration, and cost optimization services",
        "• <b>Cybersecurity & Compliance:</b> Comprehensive security assessments, implementations, and regulatory compliance services",
        "• <b>AI & Automation:</b> AI strategy, implementation, and intelligent automation solutions",
        "• <b>Application Modernization:</b> Legacy application transformation and containerization",
        "• <b>Managed Services:</b> 24/7 IT operations, monitoring, and support"
    ]
    
    services_text = "<br/>".join(services)
    about_content.append([Paragraph(services_text, ParagraphStyle('ServicesList', parent=white_normal_style, fontSize=10, leading=14, spaceAfter=10))])
    
    # Contact section
    about_content.append([Paragraph("<b>Ready to transform your organization?</b><br/>Contact us today to discuss how we can help you achieve your goals.", 
                                    ParagraphStyle('ContactHeader', parent=white_normal_style, fontSize=11, alignment=TA_CENTER, spaceAfter=8, spaceBefore=5))])
    
    contact_details = "<b>📧 Email:</b> sales@globtierinfotech.com &nbsp;&nbsp;&nbsp; <b>🌐 Website:</b> www.globtierinfotech.com"
    about_content.append([Paragraph(contact_details, ParagraphStyle('ContactDetails', parent=white_normal_style, fontSize=10, alignment=TA_CENTER, spaceAfter=0))])
    
    # Create single table with black background containing all content
    about_table = Table(about_content, colWidths=[6.4*inch])
    about_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.black),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('TOPPADDING', (0, 0), (0, 0), 12),  # First row top padding
        ('TOPPADDING', (0, 1), (-1, -1), 2),  # Other rows minimal top padding
        ('BOTTOMPADDING', (0, 0), (-1, -2), 2),  # Most rows minimal bottom padding
        ('BOTTOMPADDING', (0, -1), (-1, -1), 12),  # Last row bottom padding
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(about_table)
    
    # User info footer - Clean format
    story.append(Spacer(1, 0.3*inch))
    user_info_text = f"<i>Report generated for {user_info.get('name', 'N/A')} at {user_info.get('company', 'N/A')} ({user_info.get('industry', 'N/A')}) on {datetime.now().strftime('%B %d, %Y')}</i>"
    story.append(Paragraph(user_info_text, ParagraphStyle('Footer', parent=normal_style, fontSize=9, textColor=colors.HexColor('#6c757d'), alignment=TA_CENTER, spaceAfter=0, spaceBefore=0)))
    
    # Build PDF
    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    finally:
        # Clean up temp files after PDF is built
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                log.warning(f'Could not delete temp file {temp_file}: {e}')


def send_email_with_pdf(recipient_email, recipient_name, pdf_buffer, quiz_title, company_name, timeline=None):
    """Send email with PDF attachment using SMTP"""
    # SMTP Configuration from environment
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_username = os.getenv('SMTP_USERNAME', '')
    smtp_password = os.getenv('SMTP_PASSWORD', '')
    smtp_from_email = os.getenv('SMTP_FROM_EMAIL', smtp_username)
    smtp_from_name = os.getenv('SMTP_FROM_NAME', 'Bromho Assessments')
    
    if not smtp_username or not smtp_password:
        raise ValueError("SMTP credentials not configured. Please set SMTP_USERNAME and SMTP_PASSWORD in .env file")
    
    # Helper function to get roadmap title based on timeline
    def get_roadmap_title_from_timeline(timeline_str):
        if not timeline_str:
            return '12-Month'
        timeline_map = {
            'Immediate (0-3 months)': '3-Month',
            'Short-term (3-6 months)': '6-Month',
            'Medium-term (6-12 months)': '12-Month',
            'Long-term (12+ months)': '18-Month',
            'Exploratory phase': '6-Month'
        }
        return timeline_map.get(timeline_str, '12-Month')
    
    # Get roadmap title
    roadmap_title = get_roadmap_title_from_timeline(timeline)
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = f"{smtp_from_name} <{smtp_from_email}>"
    msg['To'] = recipient_email
    msg['Subject'] = f"Your {quiz_title} Assessment Report"
    
    # Email body (HTML template)
    email_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
          <h2 style="color: #0d6efd;">Thank you for completing the assessment!</h2>
          
          <p>Dear {recipient_name},</p>
          
          <p>We're pleased to provide you with your personalized <strong>{quiz_title}</strong> assessment report.</p>
          
          <p>Your comprehensive report is attached as a PDF document. This report includes:</p>
          <ul>
            <li>Executive Summary</li>
            <li>Maturity Level Assessment</li>
            <li>Priority Actions with timelines</li>
            <li>{roadmap_title} Roadmap</li>
            <li>Key Recommendations and Findings</li>
            <li>Next Steps</li>
          </ul>
          
          <p>We hope this report provides valuable insights for <strong>{company_name}</strong>.</p>
          
          <p>If you have any questions or would like to discuss the findings, please don't hesitate to reach out.</p>
          
          <p style="margin-top: 30px;">
            Best regards,<br/>
            <strong>{smtp_from_name}</strong>
          </p>
          
          <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
          <p style="font-size: 12px; color: #666;">
            This is an automated email. Please do not reply to this message.
          </p>
        </div>
      </body>
    </html>
    """
    
    msg.attach(MIMEText(email_body, 'html'))
    
    # Attach PDF
    pdf_buffer.seek(0)
    attachment = MIMEBase('application', 'pdf')
    attachment.set_payload(pdf_buffer.read())
    encoders.encode_base64(attachment)
    attachment.add_header(
        'Content-Disposition',
        f'attachment; filename= {quiz_title.replace(" ", "_")}_Assessment_Report.pdf'
    )
    msg.attach(attachment)
    
    # Send email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        text = msg.as_string()
        server.sendmail(smtp_from_email, recipient_email, text)
        server.quit()
        log.info(f'Email sent successfully to {recipient_email}')
        log_to_file(f'Email sent successfully to {recipient_email}')
        return True
    except Exception as e:
        log.error(f'Failed to send email: {str(e)}')
        log_to_file(f'Failed to send email: {str(e)}')
        raise


def get_question_categories(quiz_id):
    """Map questions to categories based on quiz type"""
    category_mapping = {
        'cyber': {
            'categories': ['Identify', 'Protect', 'Detect', 'Respond', 'Recover'],
            'question_map': {
                'Identify': [0, 11, 12, 14],  # Asset inventory, Framework, CISO, Board metrics
                'Protect': [1, 2, 8, 9, 10],   # MFA, Patches, Classification, Backups, Third-party
                'Detect': [4, 5],              # EDR, Logs
                'Respond': [6, 7],             # Incident response, Training
                'Recover': [3, 9]              # Pen tests, Backups
            }
        },
        'ai-adopt': {
            'categories': ['Strategy & Governance', 'Data & Infrastructure', 'Talent & Skills', 'Operations'],
            'question_map': {
                'Strategy & Governance': [0, 3, 6, 9],  # Budget, Governance, Stakeholders, Roadmap
                'Data & Infrastructure': [2, 4, 7],     # Data quality, Compute, Monitoring
                'Talent & Skills': [5, 6],               # AI talent, Business understanding
                'Operations': [1, 8, 9]                  # Use cases, Compliance, Production roadmap
            }
        },
        'cloud': {
            'categories': ['Strategy', 'Operations', 'Security', 'Cost Management'],
            'question_map': {
                'Strategy': [0, 1, 6, 13],     # Strategy, Governance, Landing zone, Exit strategy
                'Operations': [2, 3, 4, 11, 14], # Workloads, Managed services, CI/CD, Elasticity, Sandboxes
                'Security': [8, 9],            # Security controls, Skills
                'Cost Management': [5, 10, 12] # Cost optimization, FinOps, Right-sizing
            }
        },
        'cloudsec': {
            'categories': ['Identity & Access', 'Data Protection', 'Monitoring & Detection', 'Compliance'],
            'question_map': {
                'Identity & Access': [1, 11],   # IAM, Workload identity
                'Data Protection': [2, 3, 4, 13], # Secrets, Network, Encryption, Data residency
                'Monitoring & Detection': [0, 5, 6, 7, 8], # CSPM, Scanning, Logs, Container scan, WAF
                'Compliance': [9, 10, 12, 14]   # Compliance mappings, DevSecOps, Playbooks, Benchmarks
            }
        }
    }
    
    # Default mapping for other quiz types
    default_mapping = {
        'categories': ['Governance', 'Operations', 'Security', 'Innovation'],
        'question_map': {
            'Governance': list(range(0, 4)),
            'Operations': list(range(4, 8)),
            'Security': list(range(8, 12)),
            'Innovation': list(range(12, 15))
        }
    }
    
    return category_mapping.get(quiz_id, default_mapping)


def generate_industry_benchmarks(quiz_id, industry, num_questions):
    """Generate industry benchmark data for comparison with realistic trends"""
    # Industry-specific base scores (0-5 scale) - based on typical industry maturity levels
    industry_base_scores = {
        'Technology': {'base': 3.5, 'variance': 0.4, 'growth_rate': 0.08},
        'Finance': {'base': 3.8, 'variance': 0.3, 'growth_rate': 0.06},
        'Healthcare': {'base': 3.2, 'variance': 0.4, 'growth_rate': 0.07},
        'Education': {'base': 2.8, 'variance': 0.5, 'growth_rate': 0.05},
        'Manufacturing': {'base': 3.0, 'variance': 0.4, 'growth_rate': 0.06},
        'Retail': {'base': 2.9, 'variance': 0.4, 'growth_rate': 0.07},
        'Government': {'base': 3.1, 'variance': 0.3, 'growth_rate': 0.04},
        'Non-Profit': {'base': 2.5, 'variance': 0.5, 'growth_rate': 0.05},
        'Consulting': {'base': 3.6, 'variance': 0.3, 'growth_rate': 0.06},
        'Other': {'base': 3.0, 'variance': 0.4, 'growth_rate': 0.05}
    }
    
    industry_data = industry_base_scores.get(industry, industry_base_scores['Other'])
    base_score = industry_data['base']
    variance = industry_data['variance']
    growth_rate = industry_data['growth_rate']
    
    # Quiz-specific adjustments
    quiz_adjustments = {
        'cyber': {'Finance': 0.3, 'Healthcare': 0.2, 'Government': 0.2},
        'ai-adopt': {'Technology': 0.4, 'Finance': 0.3},
        'cloud': {'Technology': 0.3, 'Finance': 0.2},
        'cloudsec': {'Finance': 0.3, 'Healthcare': 0.2}
    }
    
    adjustment = quiz_adjustments.get(quiz_id, {}).get(industry, 0.0)
    adjusted_base = base_score + adjustment
    
    # Generate category scores - use deterministic approach based on industry+quiz combination
    categories = get_question_categories(quiz_id)
    category_scores = {}
    
    # Create a seed based on industry and quiz for consistent results
    seed_string = f"{industry}_{quiz_id}"
    seed_value = sum(ord(c) for c in seed_string)
    
    for idx, category in enumerate(categories['categories']):
        # Use deterministic variation based on category position and seed
        variation = (((seed_value + idx * 17) % 100) / 100 - 0.5) * variance
        category_score = adjusted_base + variation
        category_score = max(1.5, min(4.5, category_score))  # Clamp between 1.5 and 4.5
        category_scores[category] = round(category_score, 1)
    
    # Generate overall industry average
    overall_avg = sum(category_scores.values()) / len(category_scores) if category_scores else adjusted_base
    
    # Generate realistic trend data showing gradual improvement over 7 months
    trend_data = []
    try:
        from dateutil.relativedelta import relativedelta
        use_dateutil = True
    except ImportError:
        log.warning('python-dateutil not available, using manual month calculation')
        use_dateutil = False
    
    # Get current date
    now = datetime.now()
    
    # Calculate starting score (7 months ago) - industry grows gradually
    start_score = overall_avg - (growth_rate * 6)  # 6 months of growth to reach current level
    start_score = max(2.0, start_score)  # Don't go below 2.0
    
    # Generate data for 7 months: from 6 months ago to current month
    for i in range(7):
        month_offset = 6 - i  # 6, 5, 4, 3, 2, 1, 0 (oldest to newest)
        
        if use_dateutil:
            target_date = now - relativedelta(months=month_offset)
        else:
            current_month = now.month
            current_year = now.year
            target_month = current_month - month_offset
            target_year = current_year
            
            while target_month <= 0:
                target_month += 12
                target_year -= 1
            while target_month > 12:
                target_month -= 12
                target_year += 1
            
            target_date = datetime(target_year, target_month, 1)
        
        # Calculate progressive score with realistic growth pattern
        # Growth is not perfectly linear - slight acceleration in later months
        progress = i / 6  # 0 to 1
        eased_progress = progress * progress * (3 - 2 * progress)  # Smooth step function
        month_score = start_score + (overall_avg - start_score) * eased_progress
        
        # Add very small deterministic variation for realism (not random)
        month_variation = ((seed_value + i * 7) % 10 - 5) / 100  # -0.05 to +0.05
        month_score += month_variation
        
        month_score = max(2.0, min(4.5, month_score))
        trend_data.append({
            'month': target_date.month,
            'year': target_date.year,
            'score': round(month_score, 2)
        })
    
    # Calculate percentile based on the user's industry and typical distribution
    # This creates consistent percentiles for the same industry/quiz combination
    percentile_base = 50 + ((seed_value % 30) - 15)  # 35-65 range based on seed
    percentile = max(25, min(85, percentile_base))
    
    return {
        'industry': industry,
        'overallAverage': round(overall_avg, 1),
        'categoryScores': category_scores,
        'trendData': trend_data,
        'percentile': percentile
    }


def calculate_category_scores(scores, quiz_id):
    """Calculate user scores by category"""
    categories = get_question_categories(quiz_id)
    category_scores = {}
    
    for category, question_indices in categories['question_map'].items():
        category_ratings = [scores[i] for i in question_indices if i < len(scores)]
        if category_ratings:
            category_scores[category] = round(sum(category_ratings) / len(category_ratings), 1)
    
    return category_scores


def fallback_report(prompt, assessment_data=None):
    """Fallback report when LLM is unavailable - returns structured JSON"""
    # Extract assessment info if available
    scores = []
    quiz_id = None
    if assessment_data:
        scores = assessment_data.get('ratings', [])
        quiz_id = assessment_data.get('uuid', '')
    
    # Calculate average score if available
    avg_score = 2.5  # Default
    if scores:
        avg_score = round(sum(scores) / len(scores), 1)
    
    # Determine maturity level based on score
    if avg_score < 2:
        maturity_level = "Stage 1: Awareness"
    elif avg_score < 3:
        maturity_level = "Stage 2: Active"
    elif avg_score < 4:
        maturity_level = "Stage 3: Operational"
    elif avg_score < 5:
        maturity_level = "Stage 4: Systemic"
    else:
        maturity_level = "Stage 5: Transformative"
    
    # Get quiz-specific fallback actions if quiz_id is available
    fallback_actions = {
        'cyber': [
            {"title": "Create/up-to-date asset inventory & CMDB", "description": "Establish comprehensive asset tracking system", "timeline": "Q1", "budget": "TBD"},
            {"title": "Enforce MFA on all privileged accounts", "description": "Implement multi-factor authentication for security", "timeline": "Q1", "budget": "TBD"},
            {"title": "Establish 14-day patch window policy", "description": "Create systematic patch management process", "timeline": "Q1", "budget": "TBD"}
        ],
        'cloud': [
            {"title": "Set up landing-zone with guard-rails", "description": "Establish secure cloud foundation", "timeline": "Q1", "budget": "TBD"},
            {"title": "Migrate low-risk dev/test workloads", "description": "Begin cloud migration with low-risk applications", "timeline": "Q1-Q2", "budget": "TBD"},
            {"title": "Implement cost visibility dashboard", "description": "Monitor and optimize cloud spending", "timeline": "Q1", "budget": "TBD"}
        ],
        'ai-adopt': [
            {"title": "Secure executive sponsor & dedicated budget", "description": "Get leadership buy-in and funding", "timeline": "Q1", "budget": "TBD"},
            {"title": "Build cross-functional AI governance council", "description": "Establish AI oversight and strategy", "timeline": "Q1", "budget": "TBD"},
            {"title": "Deliver data-readiness quick-scan", "description": "Assess current data infrastructure", "timeline": "Q1", "budget": "TBD"}
        ]
    }
    
    # Use quiz-specific actions if available, otherwise use generic
    priority_actions = fallback_actions.get(quiz_id, [
        {"title": "Review assessment results with leadership team", "description": "Present findings and get stakeholder alignment", "timeline": "Q1", "budget": "TBD"},
        {"title": "Prioritize actions based on business impact", "description": "Focus on high-impact, low-effort initiatives", "timeline": "Q1", "budget": "TBD"},
        {"title": "Establish foundational capabilities", "description": "Build core infrastructure and processes", "timeline": "Q1-Q2", "budget": "TBD"}
    ])
    
    # Determine next steps based on whether API key is configured
    api_key_set = bool(openai_api_key and 
                      openai_api_key.strip() and 
                      (openai_api_key or '').strip().startswith('sk-'))
    
    if api_key_set and openai_client is None:
        # API key is set but client failed to initialize
        next_steps = [
            "Check /api/health endpoint or server logs for OpenAI client initialization errors",
            "Verify OpenAI API key is valid and has sufficient credits",
            "Check if proxy settings are interfering with API connections",
            "Review assessment results with leadership team",
            "Prioritize actions based on business impact"
        ]
    elif api_key_set:
        # API key is set and client exists (shouldn't reach here, but just in case)
        next_steps = [
            "Review assessment results with leadership team",
            "Prioritize actions based on business impact",
            "Implement priority actions from the roadmap"
        ]
    else:
        # No API key set
        next_steps = [
            "Set OPENAI_API_KEY in .env file for deeper insights",
            "Review assessment results with leadership team",
            "Prioritize actions based on business impact"
        ]
    
    return {
        "executiveSummary": f"Your organisation has an average maturity score of {avg_score}/5. Focus on the lowest-scoring categories first to improve overall maturity.",
        "maturityLevel": maturity_level,
        "overallScore": avg_score,
        "priorityActions": priority_actions,
        "roadmap": {
            "q1": "Assessment review + priority action planning",
            "q2": "Initial implementation + capability building",
            "q3": "Scale successful initiatives",
            "q4": "Measure progress + refine strategy"
        },
        "recommendations": [
            "Focus on foundational capabilities first",
            "Prioritize quick wins to build momentum",
            "Establish governance and measurement frameworks"
        ],
        "keyFindings": [
            f"Current maturity level: {maturity_level}",
            "Focus needed on foundational capabilities",
            "Quick wins available through targeted initiatives"
        ],
        "nextSteps": next_steps
    }


# API Routes

# ============================================================================
# Ping/Pong Route - Simple Health Check
# ============================================================================

@bp.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint - returns PONG"""
    log.info('GET /ping')
    return 'PONG', 200, {'Content-Type': 'text/plain'}


# ============================================================================
# Assessment API Routes - Main Use Case
# ============================================================================

@bp.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint with detailed system status"""
    log.info('GET /api/health')
    log_to_file('GET /api/health')
    
    health_data = {
        'status': 'OK',
        'ts': datetime.now().isoformat(),
        'openai': {
            'client_available': openai_client is not None,
            'api_key_set': bool(openai_api_key and 
                              openai_api_key.strip() and 
                              (openai_api_key or '').strip().startswith('sk-'))
        }
    }
    
    if openai_init_error:
        health_data['openai']['init_error'] = openai_init_error
    
    return jsonify(health_data)


@bp.route('/api/analyse', methods=['POST'])
def analyse():
    """Analyze prompt using OpenAI"""
    log.info('POST /api/analyse')
    log_to_file('POST /api/analyse')
    
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            log.warning('Missing or invalid prompt')
            log_to_file('Missing or invalid prompt')
            return jsonify({
                'success': False,
                'error': 'Missing or invalid prompt'
            }), 400
        
        prompt = data['prompt']
        if not isinstance(prompt, str):
            log.warning('Prompt is not a string')
            log_to_file('Prompt is not a string')
            return jsonify({
                'success': False,
                'error': 'Prompt must be a string'
            }), 400
        
        # Log assessment request data
        log.info('=' * 80)
        log.info('ASSESSMENT REQUEST DATA')
        log.info('=' * 80)
        
        # Log form data if available
        if 'formData' in data:
            form_data = data.get('formData', {})
            log.info('FORM DATA:')
            log.info(json.dumps(form_data, indent=2))
            log_to_file(f'FORM DATA: {json.dumps(form_data)}')
        
        # Log assessment data (questions and ratings) if available
        if 'assessment' in data:
            assessment_data = data.get('assessment', {})
            log.info('ASSESSMENT DATA:')
            log.info(f"  UUID: {assessment_data.get('uuid', 'N/A')}")
            log.info(f"  Total Questions: {len(assessment_data.get('questions', []))}")
            log.info(f"  Ratings: {assessment_data.get('ratings', [])}")
            
            # Log questions with ratings
            questions_with_ratings = assessment_data.get('questionsWithRatings', [])
            if questions_with_ratings:
                log.info('  QUESTIONS WITH RATINGS:')
                for i, qr in enumerate(questions_with_ratings, 1):
                    log.info(f"    Q{i}: {qr.get('question', 'N/A')}")
                    log.info(f"        Rating: {qr.get('rating', 'N/A')} ({qr.get('ratingLabel', 'N/A')})")
            
            log_to_file(f'ASSESSMENT DATA: {json.dumps(assessment_data)}')
        
        log.info('=' * 80)
        log.info(f'Body length: {len(json.dumps(data))}')
        log_to_file(f'Body length: {len(json.dumps(data))}')
        
        # Call OpenAI
        try:
            if openai_client:
                result = call_openai(prompt)
                
                # Log the complete response
                log.info('=' * 80)
                log.info('FINAL AI REPORT RESPONSE')
                log.info('=' * 80)
                log.info(json.dumps(result, indent=2, ensure_ascii=False))
                log.info('=' * 80)
                log_to_file(f'FINAL AI REPORT: {json.dumps(result, ensure_ascii=False)}')
                
                # Also print complete assessment summary as JSON
                complete_summary = {
                    'formData': data.get('formData', {}),
                    'assessment': data.get('assessment', {}),
                    'aiReport': result
                }
                log.info('')
                log.info('=' * 80)
                log.info('COMPLETE ASSESSMENT SUMMARY (JSON)')
                log.info('=' * 80)
                print(json.dumps(complete_summary, indent=2, ensure_ascii=False))
                log.info('=' * 80)
                
                # Send response to external API if UUID is present
                assessment_data = data.get('assessment', {})
                uuid = assessment_data.get('uuid')
                
                if uuid and is_valid_uuid(uuid):
                    # Prepare forms data (lead form + budget form)
                    forms_data = data.get('formData', {})
                    
                    # Prepare response data (everything else)
                    response_data = {
                        'questionsWithRatings': assessment_data.get('questionsWithRatings', []),
                        'ratings': assessment_data.get('ratings', []),
                        'questions': assessment_data.get('questions', []),
                        'aiReport': result
                    }
                    
                    # Send to external API
                    log.info(f"Sending assessment response to external API for UUID: {uuid}")
                    external_response = send_assessment_response_to_external_api(uuid, forms_data, response_data)
                    
                    if external_response:
                        log.info("Assessment response successfully sent to external API")
                    else:
                        log.warning("Failed to send assessment response to external API")
                else:
                    log.info("No valid UUID found, skipping external API response submission")
                
                return jsonify({
                    'success': True,
                    'data': result
                })
            else:
                # Check if API key is set but client failed to initialize
                api_key_set = bool(openai_api_key and 
                                  openai_api_key.strip() and 
                                  (openai_api_key or '').strip().startswith('sk-'))
                
                if api_key_set:
                    error_details = f' Initialization error: {openai_init_error}' if openai_init_error else ''
                    log.warning(f'OpenAI client not available despite API key being set, using fallback.{error_details}')
                    log.warning('Check server startup logs or /api/health endpoint for initialization errors')
                    log_to_file(f'OpenAI client not available despite API key being set, using fallback.{error_details}')
                    
                    note = 'Fallback response - OpenAI client failed to initialize.'
                    if openai_init_error:
                        note += f' Error: {openai_init_error}'
                    note += ' Check /api/health endpoint or server logs for details.'
                else:
                    log.warning('OpenAI client not available, using fallback')
                    log_to_file('OpenAI client not available, using fallback')
                    note = 'Fallback response - OpenAI API key not configured'
                
                assessment_data = data.get('assessment', {})
                return jsonify({
                    'success': True,
                    'data': fallback_report(prompt, assessment_data),
                    'note': note,
                    'openai_status': {
                        'client_available': False,
                        'api_key_set': api_key_set,
                        'init_error': openai_init_error if api_key_set else None
                    }
                })
        except Exception as e:
            log.error(f'OpenAI error: {str(e)}')
            log_to_file(f'OpenAI error: {str(e)}')
            # Return 200 with fallback data instead of 500, so frontend doesn't show offline error
            try:
                assessment_data = data.get('assessment', {})
            except:
                assessment_data = None
            return jsonify({
                'success': True,
                'error': str(e),
                'data': fallback_report(prompt, assessment_data),
                'note': f'OpenAI API error: {str(e)}. Showing fallback report.'
            })
            
    except Exception as err:
        log.error(f'Gateway error: {str(err)}')
        log_to_file(f'Gateway error: {str(err)}')
        # Return 200 with fallback data instead of 500, so frontend doesn't show offline error
        try:
            assessment_data = request.get_json().get('assessment', {}) if request.is_json else None
        except:
            assessment_data = None
        return jsonify({
            'success': True,
            'error': f'Gateway error: {str(err)}',
            'data': fallback_report('', assessment_data),
            'note': f'Gateway error occurred. Showing fallback report.'
        })


@bp.route('/api/benchmarks', methods=['POST'])
def benchmarks():
    """Get industry benchmarks and visualization data"""
    log.info('POST /api/benchmarks')
    log_to_file('POST /api/benchmarks')
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Missing request data'
            }), 400
        
        quiz_id = data.get('quizId')
        industry = data.get('industry')
        scores = data.get('scores', [])
        
        if not quiz_id or not industry:
            return jsonify({
                'success': False,
                'error': 'Missing quizId or industry'
            }), 400
        
        # Calculate user category scores
        user_category_scores = calculate_category_scores(scores, quiz_id)
        
        # Generate industry benchmarks
        num_questions = len(scores)
        benchmark_data = generate_industry_benchmarks(quiz_id, industry, num_questions)
        
        # Calculate overall user score
        overall_user_score = round(sum(scores) / len(scores), 1) if scores else 0
        
        # Prepare visualization data
        visualization_data = {
            'userScore': overall_user_score,
            'industryAverage': benchmark_data['overallAverage'],
            'userCategoryScores': user_category_scores,
            'industryCategoryScores': benchmark_data['categoryScores'],
            'trendData': benchmark_data['trendData'],
            'percentile': benchmark_data['percentile'],
            'industry': industry,
            'quizId': quiz_id
        }
        
        log.info(f'Generated benchmark data for {industry} industry, {quiz_id} quiz')
        log_to_file(f'Generated benchmark data for {industry} industry, {quiz_id} quiz')
        
        return jsonify({
            'success': True,
            'data': visualization_data
        })
        
    except Exception as err:
        log.error(f'Benchmark error: {str(err)}')
        log_to_file(f'Benchmark error: {str(err)}')
        return jsonify({
            'success': False,
            'error': f'Server error: {str(err)}'
        }), 500


@bp.route('/api/pdf-download', methods=['POST'])
def pdf_download():
    """Generate PDF and return it for download"""
    log.info('POST /api/pdf-download')
    log_to_file('POST /api/pdf-download')
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'company', 'industry', 'quizId']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Get quiz title
        quizzes = load_quizzes()
        quiz = next((q for q in quizzes if q.get('id') == data['quizId']), None)
        quiz_title = quiz.get('title', 'Assessment') if quiz else 'Assessment'
        
        # Prepare user info (including budget fields for dynamic roadmap)
        user_info = {
            'name': data.get('name', 'N/A'),
            'email': data.get('email', 'N/A'),
            'company': data.get('company', 'N/A'),
            'industry': data.get('industry', 'N/A'),
            'timeline': data.get('timeline', 'Medium-term (6-12 months)'),
            'projectBudget': data.get('projectBudget', 'Not specified'),
            'employeeCount': data.get('employeeCount', 'N/A'),
            'annualRevenue': data.get('annualRevenue', 'N/A')
        }
        
        # Get report data
        report_data = data.get('reportData')
        if not report_data:
            return jsonify({
                'success': False,
                'error': 'Missing reportData field'
            }), 400
        
        # Get benchmark data if available
        benchmark_data = data.get('benchmarkData')
        
        # Generate PDF
        try:
            pdf_buffer = generate_pdf(report_data, user_info, quiz_title, benchmark_data)
            log.info('PDF generated successfully for download')
            log_to_file('PDF generated successfully for download')
        except Exception as e:
            log.error(f'PDF generation error: {str(e)}')
            log_to_file(f'PDF generation error: {str(e)}')
            return jsonify({
                'success': False,
                'error': f'Failed to generate PDF: {str(e)}'
            }), 500
        
        # Return PDF as download
        pdf_buffer.seek(0)
        filename = f"{quiz_title.replace(' ', '_')}_Assessment_Report.pdf"
        
        return Response(
            pdf_buffer.read(),
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Type': 'application/pdf'
            }
        )
        
    except Exception as err:
        log.error(f'PDF download endpoint error: {str(err)}')
        log_to_file(f'PDF download endpoint error: {str(err)}')
        return jsonify({
            'success': False,
            'error': f'Server error: {str(err)}'
        }), 500


@bp.route('/api/pdf-email', methods=['POST'])
def pdf_email():
    """Send PDF via email using base64 encoded PDF data"""
    log.info('POST /api/pdf-email')
    log_to_file('POST /api/pdf-email')
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'company', 'industry', 'quizId', 'pdfBase64', 'pdfFilename']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Get quiz title
        quizzes = load_quizzes()
        quiz = next((q for q in quizzes if q.get('id') == data['quizId']), None)
        quiz_title = quiz.get('title', 'Assessment') if quiz else 'Assessment'
        
        # Decode base64 PDF
        import base64
        pdf_data = base64.b64decode(data['pdfBase64'])
        pdf_buffer = BytesIO(pdf_data)
        
        # Send email with PDF
        try:
            send_email_with_pdf(
                recipient_email=data['email'],
                recipient_name=data['name'],
                pdf_buffer=pdf_buffer,
                quiz_title=quiz_title,
                company_name=data['company'],
                timeline=data.get('timeline', 'Medium-term (6-12 months)')
            )
            log.info(f'Email sent to {data["email"]}')
            log_to_file(f'Email sent to {data["email"]}')
            
            return jsonify({
                'success': True,
                'message': f'PDF sent to {data["email"]}'
            })
        except Exception as e:
            log.error(f'Email sending error: {str(e)}')
            log_to_file(f'Email sending error: {str(e)}')
            return jsonify({
                'success': False,
                'error': f'Failed to send email: {str(e)}'
            }), 500
            
    except Exception as err:
        log.error(f'PDF email endpoint error: {str(err)}')
        log_to_file(f'PDF email endpoint error: {str(err)}')
        return jsonify({
            'success': False,
            'error': f'Server error: {str(err)}'
        }), 500


@bp.route('/api/pdf', methods=['POST'])
def pdf():
    """Generate PDF and send via email. Accepts either structured JSON data or HTML content."""
    log.info('POST /api/pdf')
    log_to_file('POST /api/pdf')
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'company', 'industry', 'quizId']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Get quiz title
        quizzes = load_quizzes()
        quiz = next((q for q in quizzes if q.get('id') == data['quizId']), None)
        quiz_title = quiz.get('title', 'Assessment') if quiz else 'Assessment'
        
        # Prepare user info (including budget fields for dynamic roadmap)
        user_info = {
            'name': data.get('name', 'N/A'),
            'email': data.get('email', 'N/A'),
            'company': data.get('company', 'N/A'),
            'industry': data.get('industry', 'N/A'),
            'timeline': data.get('timeline', 'Medium-term (6-12 months)'),
            'projectBudget': data.get('projectBudget', 'Not specified'),
            'employeeCount': data.get('employeeCount', 'N/A'),
            'annualRevenue': data.get('annualRevenue', 'N/A')
        }
        
        # Determine if we have HTML content or structured JSON data
        report_data = None
        if 'htmlContent' in data:
            # Parse HTML content to structured data
            try:
                report_data = parse_web_html_to_structured_data(data['htmlContent'])
                log.info('Parsed HTML content to structured data for PDF generation')
                log_to_file('Parsed HTML content to structured data for PDF generation')
            except Exception as e:
                log.error(f'HTML parsing error: {str(e)}')
                log_to_file(f'HTML parsing error: {str(e)}')
                return jsonify({
                    'success': False,
                    'error': f'Failed to parse HTML content: {str(e)}'
                }), 400
        elif 'reportData' in data:
            # Use provided structured JSON data
            report_data = data['reportData']
        else:
            return jsonify({
                'success': False,
                'error': 'Missing reportData or htmlContent field'
            }), 400
        
        # Get benchmark data if available
        benchmark_data = data.get('benchmarkData')
        
        # Generate PDF
        try:
            pdf_buffer = generate_pdf(report_data, user_info, quiz_title, benchmark_data)
            log.info('PDF generated successfully')
            log_to_file('PDF generated successfully')
        except Exception as e:
            log.error(f'PDF generation error: {str(e)}')
            log_to_file(f'PDF generation error: {str(e)}')
            return jsonify({
                'success': False,
                'error': f'Failed to generate PDF: {str(e)}'
            }), 500
        
        # Send email with PDF
        try:
            send_email_with_pdf(
                recipient_email=data['email'],
                recipient_name=data['name'],
                pdf_buffer=pdf_buffer,
                quiz_title=quiz_title,
                company_name=data['company'],
                timeline=user_info.get('timeline', 'Medium-term (6-12 months)')
            )
            log.info(f'Email sent to {data["email"]}')
            log_to_file(f'Email sent to {data["email"]}')
            
            return jsonify({
                'success': True,
                'message': f'PDF report generated and sent to {data["email"]}'
            })
        except Exception as e:
            log.error(f'Email sending error: {str(e)}')
            log_to_file(f'Email sending error: {str(e)}')
            return jsonify({
                'success': False,
                'error': f'Failed to send email: {str(e)}'
            }), 500
            
    except Exception as err:
        log.error(f'PDF endpoint error: {str(err)}')
        log_to_file(f'PDF endpoint error: {str(err)}')
        return jsonify({
            'success': False,
            'error': f'Server error: {str(err)}'
        }), 500


@bp.route('/api/config', methods=['GET'])
def get_config():
    """Get all quiz configurations"""
    log.info('GET /api/config')
    log_to_file('GET /api/config')
    quizzes = load_quizzes()
    return jsonify(quizzes)


@bp.route('/api/config/<quiz_id>', methods=['GET'])
def get_quiz_config(quiz_id):
    """Get specific quiz configuration by UUID (external API) or string ID (local quizzes.json)"""
    log.info(f'GET /api/config/{quiz_id}')
    log_to_file(f'GET /api/config/{quiz_id}')
    
    # Check if it's a UUID - if so, fetch from external API
    if is_valid_uuid(quiz_id):
        log.info(f'UUID detected, fetching from external API: {quiz_id}')
        log_to_file(f'Fetching from external API: {quiz_id}')
        
        external_data = fetch_assessment_from_external_api(quiz_id)
        if external_data:
            quiz_config = transform_external_assessment_to_config(external_data, quiz_id)
            if quiz_config:
                return jsonify(quiz_config)
        
        return jsonify({'error': 'Assessment not found', 'uuid': quiz_id}), 404
    
    # Not a UUID - check local quizzes.json for string IDs (e.g., "cyber", "cloud")
    log.info(f'String ID detected, checking local quizzes.json: {quiz_id}')
    try:
        quizzes = load_quizzes()
        quiz_config = next((q for q in quizzes if q.get('id') == quiz_id), None)
        
        if quiz_config:
            log.info(f'Found quiz config for: {quiz_id}')
            return jsonify(quiz_config)
        else:
            log.warning(f'Quiz ID not found in local catalog: {quiz_id}')
            return jsonify({'error': 'Quiz not found', 'quiz_id': quiz_id}), 404
    except Exception as e:
        log.error(f'Error loading quiz config: {e}')
        return jsonify({'error': 'Failed to load quiz configuration', 'details': str(e)}), 500


# ============================================================================
# Static File Serving Routes
# ============================================================================

@bp.route('/')
def root():
    """Root route - returns simple message"""
    return 'PONG', 200, {'Content-Type': 'text/plain'}


@bp.route('/quiz/<quiz_id>')
def assessment_quiz(quiz_id):
    """Clean URL route for assessment quiz - serves assess.html with UUID only"""
    log.info(f'GET /quiz/{quiz_id}')
    log_to_file(f'GET /quiz/{quiz_id}')
    
    # Validate quiz_id - must be a valid UUID
    if not quiz_id or not quiz_id.strip():
        return jsonify({'error': 'Invalid quiz ID'}), 400
    
    # Strict UUID validation - only accept valid UUIDs
    if not is_valid_uuid(quiz_id):
        log.warning(f'Invalid UUID format: {quiz_id}')
        return 'Invalid URL', 400, {'Content-Type': 'text/plain'}
    
    # Read assess.html and inject quiz ID into the page
    try:
        assess_html_path = os.path.join('public', 'assess.html')
        
        # Check if file exists
        if not os.path.exists(assess_html_path):
            log.error(f'Assessment file not found at: {assess_html_path}')
            return jsonify({
                'error': 'Assessment file not found',
                'path': assess_html_path
            }), 404
        
        with open(assess_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        if not html_content:
            log.error('Assessment file is empty')
            return jsonify({'error': 'Assessment file is empty'}), 500
        
        # Detect base path for deployment subpaths (e.g., /bromho-assessments)
        # Inject base path detection and API helper at the start of script section
        base_path_script = r"""
// Detect base path from current URL (handles deployment subpaths like /maturity-assessments)
const getBasePath = () => {
  const path = window.location.pathname;
  // Extract base path (everything before /quiz/)
  const match = path.match(/^(.+?)\/quiz\//);
  if (match) return match[1];
  return '';
};

const basePath = getBasePath();
const apiBase = basePath ? `${basePath}/api` : '/api';
const logoPath = basePath ? `${basePath}/download-removebg-preview.jpg` : '/download-removebg-preview.jpg';

// Debug logging (can be removed in production)
console.log('Base path detected:', basePath);
console.log('API base URL:', apiBase);

// Update logo src if base path is detected
if (basePath) {
  const logoImgs = document.querySelectorAll('.logo-white-space');
  logoImgs.forEach(img => {
    if (img) img.src = logoPath;
  });
}

// Helper function to make API calls with correct base path
const apiCall = (endpoint, options = {}) => {
  const url = endpoint.startsWith('/') ? `${apiBase}${endpoint}` : `${apiBase}/${endpoint}`;
  return fetch(url, options);
};
"""
        
        # Find and replace the query parameter reading section
        # Use regex to find the pattern more flexibly (re already imported at top)
        
        # Escape quiz_id for JavaScript (prevent XSS and syntax errors)
        import json
        quiz_id_escaped = json.dumps(quiz_id)  # This properly escapes quotes and special chars
        
        # Pattern to match: const params = ... const quizId = params.get('q');
        # Match across multiple lines
        pattern = r"const\s+params\s*=\s*new\s+URLSearchParams\([^)]+\);\s*\n\s*const\s+quizId\s*=\s*params\.get\(['\"]q['\"]\);"
        
        replacement = f"{base_path_script}\nconst quizId = {quiz_id_escaped}; // Injected from clean URL route"
        
        # Try multiline pattern first
        if re.search(pattern, html_content, re.MULTILINE):
            html_content = re.sub(pattern, replacement, html_content, flags=re.MULTILINE)
        else:
            # Try single line pattern
            pattern_single = r"const\s+params\s*=\s*new\s+URLSearchParams\([^)]+\);\s*const\s+quizId\s*=\s*params\.get\(['\"]q['\"]\);"
            if re.search(pattern_single, html_content):
                html_content = re.sub(pattern_single, replacement, html_content)
            else:
                # Try just the quizId line
                simple_pattern = r"const\s+quizId\s*=\s*params\.get\(['\"]q['\"]\);"
                if re.search(simple_pattern, html_content):
                    html_content = re.sub(simple_pattern, replacement, html_content)
                else:
                    # Last resort: inject after first <script> tag
                    html_content = re.sub(
                        r'(<script[^>]*>)',
                        r'\1\n' + replacement + '\n',
                        html_content,
                        count=1
                    )
                    log.warning(f'Could not find quizId pattern in assess.html, injected at script start. Quiz ID: {quiz_id}')
        
        # Also update the redirect to use clean URL format with base path
        # Note: basePath is a JavaScript variable, so we use double curly braces to escape in f-string
        html_content = html_content.replace(
            "if (!quizId) location.href = 'index.html';",
            "if (!quizId) location.href = basePath ? `${basePath}/assessment` : '/assessment';"
        )
        
        # Update back button to use clean URL with base path
        html_content = html_content.replace(
            "location.href='index.html'",
            "location.href = basePath ? `${basePath}/assessment` : '/assessment';"
        )
        
        # Replace all API fetch calls to use apiBase variable
        # (re already imported at module level)
        # Replace fetch('/api/... with fetch(`${apiBase}/...
        # Handle string literals: fetch('/api/config')
        html_content = re.sub(
            r"fetch\(['\"]\/api\/([^'\"]+)['\"]",
            r'fetch(`${apiBase}/\1`)',
            html_content
        )
        
        # Handle template literals: fetch(`/api/config/${quizId}`)
        # This pattern matches the entire template literal
        html_content = re.sub(
            r"fetch\(`\/api\/([^`]+)`",
            r'fetch(`${apiBase}/\1`)',
            html_content
        )
        
        # Additional fallback: replace any remaining '/api/' in fetch calls
        # This catches edge cases where the regex might miss
        html_content = re.sub(
            r"(fetch\([^)]*)\/api\/([^)]*\))",
            r'\1${apiBase}/\2',
            html_content
        )
        
        # Replace all hardcoded URLs with relative URLs (supports any deployment)
        html_content = html_content.replace(
            'https://assessments.botgo.io/api/',
            '/api/'
        )
        html_content = html_content.replace(
            'http://localhost:5000/api/',
            '/api/'
        )
        html_content = html_content.replace(
            'http://localhost:7000/api/',
            '/api/'
        )
        # Generic replacement for any localhost port
        html_content = re.sub(
            r'http://localhost:\d+/api/',
            '/api/',
            html_content
        )
        
        # Validate that quizId was injected
        if f"const quizId = {quiz_id_escaped}" not in html_content and f"const quizId = '{quiz_id}'" not in html_content:
            log.warning(f'Quiz ID may not have been properly injected. Quiz ID: {quiz_id}')
        
        log.info(f'Successfully prepared assessment page for quiz: {quiz_id}')
        return Response(html_content, mimetype='text/html', headers={'Cache-Control': 'no-cache'})
    except FileNotFoundError as e:
        log.error(f'Assessment file not found: {e}')
        return jsonify({
            'error': 'Assessment file not found',
            'details': str(e),
            'path': assess_html_path
        }), 404
    except Exception as e:
        log.error(f'Error serving assessment quiz: {e}')
        log.error(f'Exception type: {type(e).__name__}')
        import traceback
        log.error(f'Traceback: {traceback.format_exc()}')
        return jsonify({
            'error': 'Failed to load assessment',
            'details': str(e),
            'type': type(e).__name__
        }), 500


@bp.route('/assessment/quiz/<quiz_id>')
def assessment_quiz_string_id(quiz_id):
    """Route for assessment quiz with string ID (e.g., cyber, cloud) - serves assess.html"""
    log.info(f'GET /assessment/quiz/{quiz_id}')
    log_to_file(f'GET /assessment/quiz/{quiz_id}')
    
    # Validate quiz_id
    if not quiz_id or not quiz_id.strip():
        return jsonify({'error': 'Invalid quiz ID'}), 400
    
    # Check if quiz exists in quizzes.json
    try:
        with open(QUIZZES_PATH, 'r', encoding='utf-8') as f:
            quizzes = json.load(f)
        quiz_exists = any(q.get('id') == quiz_id for q in quizzes)
        if not quiz_exists:
            log.warning(f'Quiz ID not found in catalog: {quiz_id}')
            return jsonify({'error': 'Quiz not found', 'quiz_id': quiz_id}), 404
    except Exception as e:
        log.warning(f'Could not validate quiz ID: {e}')
        # Continue anyway - let the frontend handle it
    
    # Read assess.html and inject quiz ID into the page (same logic as UUID route)
    try:
        assess_html_path = os.path.join('public', 'assess.html')
        
        if not os.path.exists(assess_html_path):
            log.error(f'Assessment file not found at: {assess_html_path}')
            return jsonify({
                'error': 'Assessment file not found',
                'path': assess_html_path
            }), 404
        
        with open(assess_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        if not html_content:
            log.error('Assessment file is empty')
            return jsonify({'error': 'Assessment file is empty'}), 500
        
        # Detect base path for deployment subpaths
        base_path_script = r"""
// Detect base path from current URL (handles deployment subpaths like /maturity-assessments)
const getBasePath = () => {
  const path = window.location.pathname;
  // Extract base path (everything before /assessment/quiz/)
  const match = path.match(/^(.+?)\/assessment\/quiz\//);
  if (match) return match[1];
  return '';
};

const basePath = getBasePath();
const apiBase = basePath ? `${basePath}/api` : '/api';
const logoPath = basePath ? `${basePath}/download-removebg-preview.jpg` : '/download-removebg-preview.jpg';

// Debug logging (can be removed in production)
console.log('Base path detected:', basePath);
console.log('API base URL:', apiBase);

// Update logo src if base path is detected
if (basePath) {
  const logoImgs = document.querySelectorAll('.logo-white-space');
  logoImgs.forEach(img => {
    if (img) img.src = logoPath;
  });
}

// Helper function to make API calls with correct base path
const apiCall = (endpoint, options = {}) => {
  const url = endpoint.startsWith('/') ? `${apiBase}${endpoint}` : `${apiBase}/${endpoint}`;
  return fetch(url, options);
};
"""
        
        # Escape quiz_id for JavaScript
        import json
        quiz_id_escaped = json.dumps(quiz_id)
        
        # Pattern to match: const params = ... const quizId = params.get('q');
        pattern = r"const\s+params\s*=\s*new\s+URLSearchParams\([^)]+\);\s*\n\s*const\s+quizId\s*=\s*params\.get\(['\"]q['\"]\);"
        
        replacement = f"{base_path_script}\nconst quizId = {quiz_id_escaped}; // Injected from clean URL route"
        
        # Try multiline pattern first
        if re.search(pattern, html_content, re.MULTILINE):
            html_content = re.sub(pattern, replacement, html_content, flags=re.MULTILINE)
        else:
            # Try single line pattern
            pattern_single = r"const\s+params\s*=\s*new\s+URLSearchParams\([^)]+\);\s*const\s+quizId\s*=\s*params\.get\(['\"]q['\"]\);"
            if re.search(pattern_single, html_content):
                html_content = re.sub(pattern_single, replacement, html_content)
            else:
                # Try just the quizId line
                simple_pattern = r"const\s+quizId\s*=\s*params\.get\(['\"]q['\"]\);"
                if re.search(simple_pattern, html_content):
                    html_content = re.sub(simple_pattern, replacement, html_content)
                else:
                    # Last resort: inject after first <script> tag
                    html_content = re.sub(
                        r'(<script[^>]*>)',
                        r'\1\n' + replacement + '\n',
                        html_content,
                        count=1
                    )
                    log.warning(f'Could not find quizId pattern in assess.html, injected at script start. Quiz ID: {quiz_id}')
        
        # Update redirects to use clean URL format with base path
        html_content = html_content.replace(
            "if (!quizId) location.href = 'index.html';",
            "if (!quizId) location.href = basePath ? `${basePath}/assessment` : '/assessment';"
        )
        
        html_content = html_content.replace(
            "location.href='index.html'",
            "location.href = basePath ? `${basePath}/assessment` : '/assessment';"
        )
        
        # Replace all API fetch calls to use apiBase variable
        html_content = re.sub(
            r"fetch\(['\"]\/api\/([^'\"]+)['\"]",
            r'fetch(`${apiBase}/\1`)',
            html_content
        )
        
        html_content = re.sub(
            r"fetch\(`\/api\/([^`]+)`",
            r'fetch(`${apiBase}/\1`)',
            html_content
        )
        
        html_content = re.sub(
            r"(fetch\([^)]*)\/api\/([^)]*\))",
            r'\1${apiBase}/\2',
            html_content
        )
        
        # Replace hardcoded URLs with relative URLs
        html_content = html_content.replace(
            'https://assessments.botgo.io/api/',
            '/api/'
        )
        html_content = html_content.replace(
            'http://localhost:5000/api/',
            '/api/'
        )
        html_content = html_content.replace(
            'http://localhost:7000/api/',
            '/api/'
        )
        html_content = re.sub(
            r'http://localhost:\d+/api/',
            '/api/',
            html_content
        )
        
        log.info(f'Successfully prepared assessment page for quiz: {quiz_id}')
        return Response(html_content, mimetype='text/html', headers={'Cache-Control': 'no-cache'})
    except FileNotFoundError as e:
        log.error(f'Assessment file not found: {e}')
        return jsonify({
            'error': 'Assessment file not found',
            'details': str(e),
            'path': assess_html_path
        }), 404
    except Exception as e:
        log.error(f'Error serving assessment quiz: {e}')
        log.error(f'Exception type: {type(e).__name__}')
        import traceback
        log.error(f'Traceback: {traceback.format_exc()}')
        return jsonify({
            'error': 'Failed to load assessment',
            'details': str(e),
            'type': type(e).__name__
        }), 500


@bp.route('/assessment')
def serve_assessment_catalog():
    """Serve the assessment catalog page (index.html)"""
    log.info('GET /assessment - serving catalog')
    return send_from_directory('public', 'index.html')

@bp.route('/assessment/<path:path>')
def serve_assessment_static(path):
    """Serve static files for assessment application"""
    # Don't serve assess.html here - use /assessment/quiz/<quiz_id> instead
    if path == 'assess.html':
        return jsonify({
            'error': 'Use /maturity-assessments/quiz/<quiz_id> format',
            'example': '/maturity-assessments/quiz/ai-adopt'
        }), 404
    return send_from_directory('public', path)


@bp.route('/<path:path>')
def serve_static(path):
    """Serve static files from public directory (fallback for other paths)"""
    # Don't serve index.html here to avoid conflicts
    if path == 'index.html':
        return jsonify({'error': 'Use /maturity-assessments/quiz/<quiz_id> to access the application'}), 404
    return send_from_directory('public', path)


# Register the blueprint with the app
app.register_blueprint(bp)


if __name__ == '__main__':
    log.info('===== GitOps Deployment Successful Server =====')
    log_to_file('===== GitOps Deployment Successful Server =====')
    log.info(f'Starting Flask server on port {PORT}')
    log_to_file(f'Starting Flask server on port {PORT}')
    app.run(host='0.0.0.0', port=PORT, debug=True)