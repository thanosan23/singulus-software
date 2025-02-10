from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from models import Participant
import json

STAI_QUESTIONS = {
    "1": "I feel calm",
    "2": "I feel secure",
    "3": "I am tense",
    "4": "I am strained",
    "5": "I am at ease",
    "6": "I feel upset",
    "7": "I am presently worrying over possible misfortunes",
    "8": "I feel satisfied",
    "9": "I feel frightened",
    "10": "I feel uncomfortable",
    "11": "I feel self confident",
    "12": "I feel nervous",
    "13": "I feel jittery",
    "14": "I feel indecisive",
    "15": "I am relaxed",
    "16": "I feel content",
    "17": "I am worried",
    "18": "I feel confused",
    "19": "I feel steady",
    "20": "I feel pleasant",
}

PSS_QUESTIONS = {
    "1": "In the last month, how often have you been upset because of something that happened unexpectedly?",
    "2": "In the last month, how often have you felt that you were unable to control the important things in your life?",
    "3": "In the last month, how often have you felt nervous and stressed?",
    "4": "In the last month, how often have you felt confident about your ability to handle your personal problems?",
    "5": "In the last month, how often have you felt that things were going your way?",
    "6": "In the last month, how often have you found that you could not cope with all the things that you had to do?",
    "7": "In the last month, how often have you been able to control irritations in your life?",
    "8": "In the last month, how often have you felt that you were on top of things?",
    "9": "In the last month, how often have you been angered because of things that happened that were outside of your control?",
    "10": "In the last month, how often have you felt difficulties were piling up so high that you could not overcome them?",

}

POMS_QUESTIONS = {
    "tension": ["Tense", "Shaky", "On edge"],
    "depression": ["Sad", "Unworthy", "Discouraged"],
    "anger": ["Angry", "Grumpy", "Annoyed"],
    "vigor": ["Lively", "Active", "Energetic"],
    "fatigue": ["Worn out", "Fatigued", "Exhausted"],
}

class SurveyType(Enum):
    PRE = "pre"
    POST = "post"
    DAILY = "daily"
    CRISIS = "crisis"

@dataclass
class SurveyResponse:
    participant_id: int
    survey_type: SurveyType
    timestamp: str
    responses: Dict[str, any]

@dataclass
class Survey:
    name: str
    type: SurveyType
    questions: List[Dict[str, any]]

class SurveyManager:
    def __init__(self):
        self.surveys = self._initialize_surveys()
        self.responses: List[SurveyResponse] = []
    
    def _initialize_surveys(self) -> Dict[str, Survey]:
        return {
            "pre": Survey(
                name="Pre-Experiment Survey",
                type=SurveyType.PRE,
                questions=[
                    {
                        "id": "pre_1",
                        "text": "How would you rate your current stress level? (1-7)",
                        "type": "likert_7"
                    },
                    {
                        "id": "pre_2",
                        "text": "How comfortable are you with confined spaces? (1-7)",
                        "type": "likert_7"
                    },
                    {
                        "id": "pre_3",
                        "text": "How would you rate your ability to work in teams? (1-7)",
                        "type": "likert_7"
                    },
                    {
                        "id": "pre_4",
                        "text": "Do you have any previous experience in similar situations?",
                        "type": "open_ended"
                    },
                    {
                        "id": "pre_5",
                        "text": "What are your main concerns about the experiment?",
                        "type": "open_ended"
                    }
                ]
            ),
            "post": Survey(
                name="Post-Experiment Survey",
                type=SurveyType.POST,
                questions=[
                    {
                        "id": "post_1",
                        "text": "How would you rate your final stress level? (1-7)",
                        "type": "likert_7"
                    },
                    {
                        "id": "post_2",
                        "text": "How satisfied were you with the group dynamics? (1-7)",
                        "type": "likert_7"
                    },
                    {
                        "id": "post_3",
                        "text": "Did the confined space affect your performance? How?",
                        "type": "open_ended"
                    },
                    {
                        "id": "post_4",
                        "text": "What coping strategies did you develop?",
                        "type": "open_ended"
                    },
                    {
                        "id": "post_5",
                        "text": "Would you participate in a similar experiment again? Why?",
                        "type": "open_ended"
                    }
                ]
            ),
            "crisis": Survey(
                name="Crisis Response Survey",
                type=SurveyType.CRISIS,
                questions=[
                    {
                        "id": "crisis_1",
                        "text": "How stressful was the crisis situation? (1-7)",
                        "type": "likert_7"
                    },
                    {
                        "id": "crisis_2",
                        "text": "How effectively did the group respond? (1-7)",
                        "type": "likert_7"
                    },
                    {
                        "id": "crisis_3",
                        "text": "Describe your main emotional response to the crisis:",
                        "type": "open_ended"
                    }
                ]
            )
        }
    
    def get_survey(self, survey_type: SurveyType) -> Optional[Survey]:
        return self.surveys.get(survey_type.value)
    
    def format_for_gpt(self, survey: Survey) -> str:
        """Format survey questions for GPT prompt"""
        questions = []
        for q in survey.questions:
            if q["type"] == "likert_7":
                questions.append(f"{q['text']} (Please respond with a number 1-7)")
            else:
                questions.append(f"{q['text']} (Please provide a detailed response)")
        return "\n".join(questions)
    
    def parse_responses(self, responses: Dict[str, str], survey: Survey) -> Dict[str, any]:
        """Parse and validate survey responses"""
        parsed = {}
        for q in survey.questions:
            response = responses.get(q["text"])
            if q["type"] == "likert_7":
                try:
                    value = int(response)
                    if 1 <= value <= 7:
                        parsed[q["id"]] = value
                    else:
                        parsed[q["id"]] = None
                except:
                    parsed[q["id"]] = None
            else:
                parsed[q["id"]] = response if response else None
        return parsed

# Helper function to analyze survey results
def analyze_survey_results(responses: List[Dict[str, any]], survey: Survey) -> Dict[str, any]:
    analysis = {
        "survey_name": survey.name,
        "total_responses": len(responses),
        "metrics": {}
    }
    
    for q in survey.questions:
        if q["type"] == "likert_7":
            values = [r[q["id"]] for r in responses if r[q["id"]] is not None]
            if values:
                analysis["metrics"][q["id"]] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
    
    return analysis
