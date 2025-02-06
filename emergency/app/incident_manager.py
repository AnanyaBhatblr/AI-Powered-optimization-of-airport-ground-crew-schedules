import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from datetime import datetime


class IncidentManager:
    def __init__(self):
        # Load BERT model for incident analysis
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

        # Severity categories
        self.severity_levels = {
            1: "Low",
            2: "Medium",
            3: "High",
            4: "Critical"
        }

        self.incidents = pd.DataFrame()
        self.process_analyzer = None  # Will be set from app.py

    def set_process_analyzer(self, analyzer):
        """Set process analyzer instance"""
        self.process_analyzer = analyzer

    def add_incident(self, incident_data):
        """Add new incident and update tracking"""
        # Ensure required fields
        if 'Status' not in incident_data:
            incident_data['Status'] = 'Active'
        if 'Timestamp' not in incident_data:
            incident_data['Timestamp'] = pd.Timestamp.now()

        # Add to incidents DataFrame
        self.incidents = pd.concat(
            [self.incidents, pd.DataFrame([incident_data])], ignore_index=True)

        # Update process analyzer
        if self.process_analyzer:
            self.process_analyzer.add_incident(incident_data)

    def get_active_incidents(self):
        """Get all unresolved incidents"""
        return self.incidents[self.incidents['Status'] == 'Active']

    def analyze_incident_report(self, report_text):
        """Enhanced incident analysis"""
        # Get BERT embeddings for better text understanding
        inputs = self.tokenizer(
            report_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Classify severity
        severity = self._classify_severity(report_text)

        # Extract key information
        incident_info = {
            'severity': severity,
            'keywords': self._extract_keywords(report_text),
            'required_skills': self._determine_required_skills(report_text),
            'estimated_duration': self._estimate_duration(report_text)
        }

        return incident_info

    def _classify_severity(self, text):
        """Simple rule-based severity classification"""
        critical_keywords = ['fire', 'crash', 'emergency', 'injury', 'hazard']
        high_keywords = ['delay', 'equipment failure', 'weather', 'security']
        medium_keywords = ['maintenance', 'scheduling', 'staffing']

        text = text.lower()

        if any(word in text for word in critical_keywords):
            return 4
        elif any(word in text for word in high_keywords):
            return 3
        elif any(word in text for word in medium_keywords):
            return 2
        return 1

    def _extract_keywords(self, text):
        """
        Extract relevant keywords from incident report text
        """
        # Common airport operations related keywords
        operation_keywords = {
            'equipment': ['vehicle', 'truck', 'cart', 'forklift', 'loader', 'machinery'],
            'infrastructure': ['runway', 'taxiway', 'gate', 'terminal', 'apron', 'hangar'],
            'weather': ['rain', 'snow', 'ice', 'storm', 'wind', 'lightning', 'fog'],
            'emergency': ['fire', 'crash', 'injury', 'medical', 'hazard', 'accident'],
            'maintenance': ['repair', 'inspection', 'cleaning', 'servicing', 'breakdown'],
            'operations': ['delay', 'congestion', 'backup', 'scheduling', 'staffing']
        }

        text = text.lower()
        found_keywords = []

        # Search for keywords in each category
        for category, words in operation_keywords.items():
            for word in words:
                if word in text:
                    found_keywords.append({
                        'category': category,
                        'keyword': word
                    })

        return found_keywords

    def _estimate_duration(self, text):
        """
        Estimate incident resolution duration in minutes based on description
        """
        # Default duration for different severity levels
        duration_by_severity = {
            4: 120,  # Critical: 2 hours
            3: 90,   # High: 1.5 hours
            2: 60,   # Medium: 1 hour
            1: 30    # Low: 30 minutes
        }

        severity = self._classify_severity(text)
        return duration_by_severity[severity]

    def _determine_required_skills(self, text):
        """
        Determine required skills based on incident description
        """
        skill_requirements = {
            'technical': ['repair', 'maintenance', 'equipment', 'system', 'mechanical', 'electrical'],
            'emergency': ['fire', 'medical', 'security', 'hazard', 'injury', 'crash'],
            'operations': ['scheduling', 'coordination', 'logistics', 'planning'],
            'specialist': ['inspection', 'certification', 'assessment', 'audit']
        }

        text = text.lower()
        required_skills = []

        for skill_type, keywords in skill_requirements.items():
            if any(keyword in text for keyword in keywords):
                required_skills.append(skill_type)

        return required_skills if required_skills else ['general']
