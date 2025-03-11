import json
import requests
from enum import IntEnum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


# Define enums for disease classifications
class DiseaseType(IntEnum):
    NOT_A_DISEASE = 0  # Added new category for non-disease inputs
    PRIMARY_TREATABLE = 1
    SYMPTOMS_COMPLICATIONS = 2
    STRUCTURAL_CONDITIONS = 3
    INTERVENTIONS_PROCEDURES = 4
    OTHER_MEDICAL_ENTITIES = 5
    
    @classmethod
    def get_name(cls, value: int) -> str:
        names = {
            0: "Not A Medical Condition",  # Name for non-disease category
            1: "Primary Treatable Diseases",
            2: "Symptoms/Complications",
            3: "Structural Conditions",
            4: "Interventions/Procedures",
            5: "Other Medical Entities"
        }
        return names.get(value, "Unknown")


class TherapeuticArea(IntEnum):
    ONCOLOGY = 1
    CARDIOVASCULAR = 2
    NEUROLOGY_CNS = 3
    INFECTIOUS_DISEASES = 4
    METABOLIC_ENDOCRINE = 5
    IMMUNOLOGY_INFLAMMATION = 6
    RESPIRATORY = 7
    GASTROENTEROLOGY = 8
    DERMATOLOGY = 9
    NEPHROLOGY_UROLOGY = 10
    HEMATOLOGY = 11
    MUSCULOSKELETAL = 12
    OPHTHALMOLOGY = 13
    PSYCHIATRY_MENTAL_HEALTH = 14
    RARE_DISEASES = 15
    UNCLASSIFIED = 16
    NOT_APPLICABLE = 17  # Added for non-disease inputs
    
    @classmethod
    def get_name(cls, value: int) -> str:
        names = {
            1: "Oncology",
            2: "Cardiovascular",
            3: "Neurology/CNS",
            4: "Infectious Diseases",
            5: "Metabolic/Endocrine",
            6: "Immunology/Inflammation",
            7: "Respiratory",
            8: "Gastroenterology",
            9: "Dermatology",
            10: "Nephrology/Urology",
            11: "Hematology",
            12: "Musculoskeletal",
            13: "Ophthalmology",
            14: "Psychiatry/Mental Health",
            15: "Rare Diseases",
            16: "Unclassified/Other",
            17: "Not Applicable"  # Name for the new category
        }
        return names.get(value, "Unknown")


# Define Pydantic model for disease classification response
class DiseaseClassification(BaseModel):
    disease_type: int = Field(..., ge=0, le=5)  # Updated to allow 0 for NOT_A_DISEASE
    disease_type_name: str
    therapeutic_area: int = Field(..., ge=1, le=17)  # Updated to allow 17 for NOT_APPLICABLE
    therapeutic_area_name: str
    reasoning: str
    confidence: float = Field(..., ge=0, le=1)
    
    # Updated to use field_validator instead of validator
    @field_validator('disease_type_name')
    @classmethod
    def validate_disease_type_name(cls, v, info):
        values = info.data
        if 'disease_type' in values:
            expected_name = DiseaseType.get_name(values['disease_type'])
            if v != expected_name:
                raise ValueError(f"Expected disease_type_name to be {expected_name}")
        return v
    
    # Updated to use field_validator instead of validator
    @field_validator('therapeutic_area_name')
    @classmethod
    def validate_therapeutic_area_name(cls, v, info):
        values = info.data
        if 'therapeutic_area' in values:
            expected_name = TherapeuticArea.get_name(values['therapeutic_area'])
            if v != expected_name:
                raise ValueError(f"Expected therapeutic_area_name to be {expected_name}")
        return v
    
    # Updated to use field_validator instead of validator
    @field_validator('therapeutic_area')
    @classmethod
    def validate_therapeutic_area(cls, v, info):
        values = info.data
        # If it's NOT_A_DISEASE, therapeutic area should be NOT_APPLICABLE
        if 'disease_type' in values and values['disease_type'] == DiseaseType.NOT_A_DISEASE and v != TherapeuticArea.NOT_APPLICABLE:
            raise ValueError("Non-disease inputs must have therapeutic_area set to 17 (Not Applicable)")
        # If it's not Primary Treatable, therapeutic area should be UNCLASSIFIED (unless NOT_A_DISEASE)
        elif 'disease_type' in values and values['disease_type'] != DiseaseType.PRIMARY_TREATABLE and values['disease_type'] != DiseaseType.NOT_A_DISEASE and v != TherapeuticArea.UNCLASSIFIED:
            raise ValueError("Non-Primary Treatable Diseases must have therapeutic_area set to 16 (Unclassified/Other)")
        return v


# Define the prompt template
DISEASES_PROMPT = """
You are a medical classification expert. Classify the following disease/medical condition:
Disease ID: {disease_id}
Disease/condition name: {disease_name}

FIRST STEP - Determine if this is a medical condition at all:
If the input is NOT a medical condition or disease (e.g., foods, animals, general objects, activities), 
classify it as category 0 - "Not A Medical Condition".

If it IS a medical condition, then classify according to the types below:
DISEASE TYPE CLASSIFICATION:
0. Not A Medical Condition - Input is not a disease or medical condition at all (e.g., "ostrich meat", "football", "sunrise")
1. Primary Treatable Diseases - Medical conditions (including congenital ones) that are direct targets for drug therapy (e.g., autoimmune diabetes, neonatal diabetes)
2. Symptoms/Complications - Clinical manifestations or secondary effects of diseases (e.g., loss of awareness, diabetic coma)
3. Structural Conditions - Physical or anatomical abnormalities primarily addressed through surgical intervention rather than medication (e.g., cleft palate, certain heart defects)
4. Interventions/Procedures - Medical interventions or procedures (e.g., amputation, transplantation)
5. Other Medical Entities - Medical entities that don't fit clearly into the above categories

IMPORTANT: After determining the above classification, proceed based on the type:
- If category 0 (Not A Medical Condition): set therapeutic_area to 17 (Not Applicable)
- If category 1 (Primary Treatable Disease): classify into one of the therapeutic areas below (1-15)
- For all other categories (2-5): set therapeutic_area to 16 (Unclassified/Other)

THERAPEUTIC AREA CLASSIFICATION (For Primary Treatable Diseases ONLY):
1. Oncology - All types of cancers and related conditions
2. Cardiovascular - Heart diseases, stroke, hypertension, arrhythmias
3. Neurology/CNS - Neurological disorders, Alzheimer's, Parkinson's, epilepsy, multiple sclerosis
4. Infectious Diseases - Bacterial, viral, fungal infections, including HIV, tuberculosis, hepatitis
5. Metabolic/Endocrine - Diabetes, thyroid disorders, metabolic syndrome, obesity
6. Immunology/Inflammation - Autoimmune diseases, allergies, rheumatoid arthritis, psoriasis
7. Respiratory - Asthma, COPD, pulmonary fibrosis, pneumonia
8. Gastroenterology - IBD, IBS, GERD, liver diseases, pancreatic disorders
9. Dermatology - Skin conditions, eczema, acne, dermatitis
10. Nephrology/Urology - Kidney diseases, UTIs, renal failure, bladder disorders
11. Hematology - Blood disorders, anemia, leukemia, clotting disorders
12. Musculoskeletal - Bone and joint diseases, osteoporosis, arthritis
13. Ophthalmology - Eye diseases, glaucoma, macular degeneration
14. Psychiatry/Mental Health - Depression, anxiety, schizophrenia, bipolar disorder
15. Rare Diseases - Orphan diseases, genetic disorders
16. Unclassified/Other - For non-Primary Treatable Diseases
17. Not Applicable - For inputs that are not medical conditions at all

Respond in valid JSON format with the following structure:

If the input is NOT a medical condition:
{{
    "disease_type": 0,
    "disease_type_name": "Not A Medical Condition",
    "therapeutic_area": 17,
    "therapeutic_area_name": "Not Applicable",
    "reasoning": "This input is not a medical condition because...",
    "confidence": 0.9
}}

If the input is a Primary Treatable Disease:
{{
    "disease_type": 1,
    "disease_type_name": "Primary Treatable Diseases",
    "therapeutic_area": 6,  // Example: value 1-15 as appropriate
    "therapeutic_area_name": "Immunology/Inflammation",  // Corresponding name
    "reasoning": "Brief explanation for your classification",
    "confidence": 0.9
}}

If the input is a medical condition but NOT a Primary Treatable Disease:
{{
    "disease_type": 2,  // or 3, 4, 5 as appropriate
    "disease_type_name": "Symptoms/Complications",  // or the appropriate name
    "therapeutic_area": 16,
    "therapeutic_area_name": "Unclassified/Other",
    "reasoning": "Brief explanation for your classification",
    "confidence": 0.9
}}

IMPORTANT:
1. DO NOT return lists/arrays for any values - use single values only
2. For disease_type, use a single integer (0-5)
3. For therapeutic_area, ALWAYS use a single integer as specified above
4. For therapeutic_area_name, ALWAYS include the corresponding name for the therapeutic_area value
5. Make sure the entire JSON is valid
6. Do not include extra quotes within values
7. DO NOT use null values for any field
8. Be especially careful with items like foods, common objects, or activities - these are NOT medical conditions
"""


# Class to handle disease classification using Ollama
class DiseaseClassifier:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate"):
        self.ollama_url = ollama_url
        self.model = "deepseek-r1"
    
    def classify_disease(self, disease_id: str, disease_name: str) -> DiseaseClassification:
        """
        Classifies a disease using the Ollama LLM and returns a structured response.
        
        Args:
            disease_id (str): The ID of the disease
            disease_name (str): The name of the disease
            
        Returns:
            DiseaseClassification: A Pydantic model with the classification results
        """
        # Format the prompt
        prompt = DISEASES_PROMPT.format(disease_id=disease_id, disease_name=disease_name)
        
        # Prepare the request to Ollama
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1  # Low temperature for more deterministic results
            }
        }
        
        try:
            # Send the request to Ollama
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            
            # Extract the generated text
            result = response.json()
            generated_text = result.get("response", "")
            
            # Extract the JSON part from the generated text
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in the response")
                
            json_text = generated_text[json_start:json_end]
            
            # Parse the JSON
            classification_data = json.loads(json_text)
            
            # Perform pre-validation fixes
            disease_type = classification_data.get("disease_type")
            
            # Handle non-disease inputs (type 0)
            if disease_type == DiseaseType.NOT_A_DISEASE:
                classification_data["therapeutic_area"] = TherapeuticArea.NOT_APPLICABLE
                classification_data["therapeutic_area_name"] = "Not Applicable"
            
            # Handle non-primary treatable disease inputs (type 2-5)
            elif disease_type != DiseaseType.PRIMARY_TREATABLE:
                classification_data["therapeutic_area"] = TherapeuticArea.UNCLASSIFIED
                classification_data["therapeutic_area_name"] = "Unclassified/Other"
                
            # Ensure therapeutic_area_name matches the therapeutic_area
            correct_area_name = TherapeuticArea.get_name(classification_data.get("therapeutic_area", 16))
            classification_data["therapeutic_area_name"] = correct_area_name
            
            # Ensure disease_type_name matches the disease_type
            correct_type_name = DiseaseType.get_name(classification_data.get("disease_type", 1))
            classification_data["disease_type_name"] = correct_type_name
            
            # Check for common non-disease items and force classification if needed
            common_non_diseases = [
                "meat", "food", "drink", "animal", "plant", "tree", "flower", "vehicle", 
                "car", "boat", "building", "furniture", "clothing", "sport", "game", 
                "movie", "book", "music", "art"
            ]
            
            if any(term in disease_name.lower() for term in common_non_diseases):
                # Double-check if it might be a non-disease
                if disease_type != DiseaseType.NOT_A_DISEASE:
                    # The model didn't classify it as a non-disease, but it probably is
                    if "confidence" in classification_data and classification_data["confidence"] < 0.8:
                        # Low confidence - override to NOT_A_DISEASE
                        classification_data["disease_type"] = DiseaseType.NOT_A_DISEASE
                        classification_data["disease_type_name"] = "Not A Medical Condition"
                        classification_data["therapeutic_area"] = TherapeuticArea.NOT_APPLICABLE
                        classification_data["therapeutic_area_name"] = "Not Applicable"
                        classification_data["reasoning"] = f"{disease_name} is likely not a medical condition but rather a common object/item."
            
            # Validate and create the Pydantic model
            return DiseaseClassification(**classification_data)
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during disease classification: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Initialize the classifier
    classifier = DiseaseClassifier()
    
    # Example diseases and non-diseases
    items = [
        {"id": "D001", "name": "Type 2 Diabetes Mellitus"},
        {"id": "D002", "name": "Diabetic Coma"},
        {"id": "D003", "name": "Cleft Palate"},
        {"id": "D004", "name": "Kidney Transplantation"},
        {"id": "D005", "name": "Rheumatoid Arthritis"},
        {"id": "D006", "name": "ostrich meat"},
        {"id": "D007", "name": "apple juice"},
        {"id": "D008", "name": "football injury"}
    ]
    
    # Classify each item
    for item in items:
        try:
            result = classifier.classify_disease(item["id"], item["name"])
            print(f"\nClassification for {item['name']}:")
            print(f"Disease Type: {result.disease_type} - {result.disease_type_name}")
            print(f"Therapeutic Area: {result.therapeutic_area} - {result.therapeutic_area_name}")
            print(f"Reasoning: {result.reasoning}")
            print(f"Confidence: {result.confidence}")
        except Exception as e:
            print(f"Error classifying {item['name']}: {str(e)}")