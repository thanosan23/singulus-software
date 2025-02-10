import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from openai import OpenAI

from torch import nn
import torch
import random
import json
from surveys import Survey 
import threading
from tqdm import tqdm

from dotenv import load_dotenv
import os

load_dotenv('.env')

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@dataclass
class PsychologicalState:
    stress: float = 0.0
    anxiety: float = 0.0
    depression: float = 0.0
    social_satisfaction: float = 0.5
    emotional_stability: float = 0.5
    cognitive_load: float = 0.0
    trauma_exposure: float = 0.0
    resilience: float = 0.7

    def to_dict(self) -> Dict:
        return {k: float(v) for k, v in self.__dict__.items()}

class AgentBrain(nn.Module):
    def __init__(self, input_size=12, hidden_size=24, output_size=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

class AIAgent:
    def __init__(self, id: int, openai_key: str):
        self.id = id
        self.openai_key = openai_key
        self.psych_state = PsychologicalState()
        self.memory = []
        self.personality = self._generate_personality()
        self.social_connections = set()
        self.resources = random.randint(50, 100)
        self.adaptation_score = random.randint(60, 90)
        self.stress_level = 0
        self.pre_survey = None
        self.survey_results = None
        self._state_lock = threading.Lock()
        self._survey_cache = {}
        self._memory_lock = threading.Lock()
        self.base_stress = random.randint(25, 35)
        self.stress_level = self.base_stress

    def _generate_personality(self) -> Dict[str, float]:
        return {
            "openness": np.random.normal(0.5, 0.15),
            "conscientiousness": np.random.normal(0.6, 0.1),
            "extroversion": np.random.normal(0.5, 0.15),
            "agreeableness": np.random.normal(0.6, 0.1),
            "neuroticism": np.random.normal(0.5, 0.15),
            "resilience": np.random.normal(0.6, 0.15),
            "trauma_sensitivity": np.random.normal(0.5, 0.2),
            "leadership_tendency": np.random.beta(2, 5)
        }

    def update_state(self, environment: Dict, social_context: Dict) -> None:
        with self._state_lock:
            if environment.get('confined'):
                base_stress_rate = 2.0
                stress_ceiling = 95
                stress_floor = 45
                social_impact = 0.4
            else:
                # Open spaces are naturally less stressful
                base_stress_rate = 1.0
                stress_ceiling = 75
                stress_floor = 25
                social_impact = 0.2
            
            # Exits affect how people perceive their environment
            if environment.get('has_exit'):
                # Having an exit gives people a sense of control, reducing stress buildup
                stress_accumulation = 0.7
                recovery_rate = 1.3
            else:
                # No exit increases feelings of being trapped
                stress_accumulation = 1.4
                recovery_rate = 0.8
                stress_floor += 10  # Being trapped raises minimum stress
            
            # Calculate real-time stress changes
            social_support = len(self.social_connections) * social_impact
            isolation_factor = (1 - len(self.social_connections)/10) * 0.5

            # Personality affects how one handles the situation
            stress_sensitivity = (
                self.personality['neuroticism'] * 0.4 +
                -self.personality['resilience'] * 0.4 +
                -self.personality['extroversion'] * 0.2 +
                self.personality['trauma_sensitivity'] * 0.3
            )

            # Calculate stress change
            environmental_pressure = base_stress_rate * stress_accumulation
            if environment.get('crisis_event'):
                environmental_pressure *= 1.5  # Crises amplify environmental stress

            stress_change = (
                environmental_pressure * stress_sensitivity +
                -social_support * recovery_rate +
                isolation_factor
            )

            # Add some randomness to represent daily variations
            stress_change += np.random.normal(0, 0.2)
            
            # Update stress level within environment-specific bounds
            self.stress_level = int(np.clip(
                self.stress_level + stress_change,
                stress_floor,
                stress_ceiling
            ))

            # Update adaptation based on how well they're handling stress
            adaptation_delta = (
                -abs(stress_change) * 0.5 +  # Negative impact from stress changes
                social_support * 1.5 +        # Positive impact from social connections
                (1 - self.stress_level/100) * 0.8  # Better adaptation when less stressed
            )

            self.adaptation_score = max(0, min(100,
                self.adaptation_score + adaptation_delta
            ))

            # Record state
            self.memory.append({
                'timestamp': len(self.memory),
                'environment': environment.copy(),
                'connections': list(self.social_connections),
                'state': self.psych_state.to_dict(),
                'stress': self.stress_level,
                'adaptation': self.adaptation_score,
                'stress_change': stress_change
            })

    def share_resources(self) -> int:
        share_willingness = (
            self.personality["agreeableness"] * 0.4 +
            (1 - self.psych_state.stress) * 0.3 +
            self.personality["extroversion"] * 0.3
        )

        max_share = int(self.resources * 0.7)
        share_amount = int(max_share * share_willingness)

        if share_amount > 0:
            self.resources -= share_amount
            return share_amount
        return 0

    def take_survey(self, survey: Survey) -> Dict[str, str]:
        cache_key = f"{self.id}_{survey.name}_{len(self.memory)}"
        if cache_key in self._survey_cache:
            return self._survey_cache[cache_key]

        try:
            # Create a more detailed prompt for GPT
            env_type = 'Confined' if any(m['environment'].get('confined') for m in self.memory[-5:]) else 'Open'
            has_exit = any(m['environment'].get('has_exit') for m in self.memory[-5:])

            prompt = f"""You are participant {self.id} in a space psychology experiment. You are in a {env_type} space {'with' if has_exit else 'without'} access to exits.

Your current state:
- Stress Level: {self.stress_level}/100
- Social Connections: {len(self.social_connections)}
- Adaptation Score: {self.adaptation_score:.1f}/100

Your personality traits:
{chr(10).join([f'- {k.title()}: {v:.2f}' for k, v in self.personality.items()])}

Please provide natural, detailed responses to these survey questions. For Likert scales (1-7), consider how your personality and current state would influence your response. For open-ended questions, provide thoughtful 2-3 sentence responses that reflect your experiences.

Survey Questions:
{chr(10).join([f"{q['id']}: {q['text']}" for q in survey.questions])}"""

            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI agent participating in a space psychology experiment. Provide realistic, psychologically consistent responses that reflect your current state and personality."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=1000)

            # Parse and format responses
            result = {}
            text = response.choices[0].message.content
            for question in survey.questions:
                qid = question["id"]
                if question["type"] == "likert_7":
                    # Extract numerical responses for Likert questions
                    for line in text.split('\n'):
                        if qid in line:
                            try:
                                num = int(''.join(filter(str.isdigit, line)))
                                if 1 <= num <= 7:
                                    result[qid] = str(num)
                                    break
                            except:
                                continue
                    if qid not in result:
                        result[qid] = str(self._generate_likert_response(question["text"]))
                else:
                    # Extract open-ended responses
                    for line in text.split('\n'):
                        if qid in line and ':' in line:
                            answer = line.split(':', 1)[1].strip()
                            if len(answer) > 10:  # Ensure we got a real response
                                result[qid] = answer
                                break
                    if qid not in result:
                        result[qid] = "Currently processing this experience..."

            with self._state_lock:
                self._survey_cache[cache_key] = result
            return result

        except Exception as e:
            print(f"Survey failed for agent {self.id}: {e}")
            return self._generate_fallback_responses(survey)

    def _create_simplified_context(self) -> str:
        """Simplified context creation to reduce computation"""
        return json.dumps({
            'stress': self.psych_state.stress,
            'social_connections': len(self.social_connections),
            'personality': {k: round(v, 2) for k, v in self.personality.items()}
        })

    def _create_survey_context(self) -> str:
        recent_experiences = [
            {
                'environment': m['environment'],
                'connections': m['connections']  # Access connections directly
            }
            for m in self.memory[-5:]
        ] if self.memory else []

        return json.dumps({
            'personality': self.personality,
            'current_state': self.psych_state.to_dict(),
            'social_connections': len(self.social_connections),
            'stress_level': self.stress_level,
            'recent_experiences': recent_experiences
        })

    def _parse_gpt_response(self, response: str, survey: Survey) -> Dict[str, str]:
        """Parse GPT response into structured survey answers"""
        responses = {}
        lines = response.strip().split('\n')

        for question in survey.questions:
            for line in lines:
                # Match questions by ID or text
                if question["id"] in line or question["text"] in line:
                    if ':' in line:
                        _, answer = line.split(':', 1)
                        responses[question["id"]] = answer.strip()
                        break

            # Validate and clean responses
            if question["id"] not in responses:
                if question["type"] == "likert_7":
                    responses[question["id"]] = str(self._generate_likert_response(question["text"]))
                else:
                    responses[question["id"]] = "Response pending"

        return responses

    def _generate_likert_response(self, question: str) -> int:
        """Generate contextually appropriate Likert responses"""
        base_value = 4  # neutral starting point

        if "stress" in question.lower():
            return max(1, min(7, int(self.stress_level / 15)))
        elif "satisfaction" in question.lower():
            return max(1, min(7, int(self.adaptation_score / 15)))
        elif "comfort" in question.lower():
            comfort = (1 - self.personality["neuroticism"]) * 7
            return max(1, min(7, int(comfort)))

        return base_value

    def _generate_fallback_responses(self, survey: Survey) -> Dict[str, str]:
        responses = {}
        for question in survey.questions:
            if question["type"] == "likert_7":
                if "stress" in question["text"].lower():
                    responses[question["id"]] = str(self.stress_level)
                else:
                    responses[question["id"]] = str(random.randint(1, 7))
            else:
                responses[question["id"]] = "No response available"
        return responses

    def _generate_fallback_response(self, question: str) -> str:
        if "stress" in question.lower():
            return str(self.stress_level)
        elif "satisfaction" in question.lower():
            return str(int(self.psych_state.social_satisfaction * 100))
        elif any(word in question.lower() for word in ["rate", "scale", "number"]):
            return str(random.randint(1, 7))
        else:
            return "No response available"
