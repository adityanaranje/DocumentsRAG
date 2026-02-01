import json
import re
from pathlib import Path
from typing import Dict, List

# Section headers for semantic chunking
SECTION_HEADERS = [
    "Overview", "About", "Introduction", "Key Features", "Features",
    "Eligibility", "Who Can Buy", "Entry Age", "Benefits", "Death Benefit",
    "Maturity Benefit", "Survival Benefit", "Riders", "Add-on", "Premium",
    "Charges", "Policy Charges", "Fund Management", "Policy Term", "Term",
    "Exclusions", "What is Not Covered", "Tax Benefits", "Tax Benefit",
    "Claim Process", "How to Claim", "Claims", "Disclaimers", "Disclaimer",
    "Sum Assured", "Cover Amount", "Waiting Period", "Free Look", "Grace Period",
    "Surrender", "Loan", "Revival", "Settlement", "Nomination"
]

# Compile regex patterns for section detection
SECTION_PATTERNS = [re.compile(rf'\b{header}\b', re.IGNORECASE) for header in SECTION_HEADERS]


class MetadataExtractor:
    """
    Extracts metadata from file paths and enriches them with an external JSON config.
    docs/Insurer/InsuranceType/Product.pdf + plan_metadata.json
    """
    
    def __init__(self, base_path: str, config_path: str = "configs/plan_metadata.json"):
        self.base_path = Path(base_path).resolve()
        self.config_path = Path(config_path)
        self.external_metadata = self._load_external_metadata()

    def _load_external_metadata(self) -> Dict[str, Dict]:
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def _generate_plan_id(self, provider: str, insurance_type: str, product_name: str) -> str:
        """
        Generate deterministic plan_id for linking brochure and CIS.
        Format: provider_insurancetype_productname (lowercase, underscores)
        """
        # Remove doc type suffixes from product name for consistent plan_id
        clean_product = re.sub(r'[\s_-]?(brochure|cis)$', '', product_name, flags=re.IGNORECASE).strip()
        normalized = f"{provider}_{insurance_type}_{clean_product}"
        return re.sub(r'[^a-z0-9]+', '_', normalized.lower()).strip('_')

    def extract_from_path(self, file_path: str) -> Dict[str, str]:
        """
        Parses the file path to extract insurer and insurance type (category) from folders.
        Enhanced with plan_id and document_type for RAG compliance.
        """
        path = Path(file_path).resolve()
        
        if not str(path).lower().startswith(str(self.base_path).lower()):
            raise ValueError(f"File {file_path} is not inside base directory {self.base_path}")

        relative_path = path.relative_to(self.base_path)
        parts = relative_path.parts
        raw_stem = path.stem

        # Initialize product_name and doc_type
        product_name = raw_stem
        doc_type = "brochure"  # Default to brochure if not specified

        # Normalize common typos in product name using regex for safety
        product_name = re.sub(r'\bEdelwiess\b', 'Edelweiss', product_name, flags=re.IGNORECASE)
        product_name = re.sub(r'Edelweis(?!s)', 'Edelweiss', product_name, flags=re.IGNORECASE)
        product_name = re.sub(r'Smapoorna', 'Sampoorna', product_name, flags=re.IGNORECASE)
        product_name = re.sub(r'Fotune', 'Fortune', product_name, flags=re.IGNORECASE)
        product_name = re.sub(r'^(Tata\s*AIA|TATA_AIA)\b', 'TATA AIA', product_name, flags=re.IGNORECASE)
        
        stem_lower = product_name.lower()
        
        # Detect document type (CIS vs Brochure)
        if "cis" in stem_lower:
            doc_type = "cis"
            product_name = re.sub(r'[\s_-]cis\b', '', product_name, flags=re.IGNORECASE).strip()
        elif "brochure" in stem_lower:
            doc_type = "brochure"
            product_name = re.sub(r'[\s_-]brochure\b', '', product_name, flags=re.IGNORECASE).strip()

        # Extract Category (Insurance Type) from Folder Structure
        # Expected structure: docs/Insurer/Category/Product.pdf
        category = "General"
        insurer = "Other"
        
        if len(parts) >= 2:
            insurer = parts[0]
            if len(parts) >= 3:
                category = parts[1]
            else:
                category = "General"

        # Standardize Categories
        category_mapping = {
            "Term Insurance Plans": "Term Insurance",
            "Term Plans": "Term Insurance",
            "Term Plan": "Term Insurance",
            "ULIP Plans": "Unit Linked Insurance Plan",
            "ULIP Plan": "Unit Linked Insurance Plan",
            "Wealth Creation Plans": "Unit Linked Insurance Plan",
            "Savings Plan": "Savings Plan",
            "Savings Plans": "Savings Plan",
            "Guaranteed Income Plans": "Savings Plan",
            "Retirement Plans": "Retirement and Pension",
            "Retirement and Pension Plan": "Retirement and Pension",
            "Pension Plan": "Retirement and Pension",
            "Health Plan": "Health Insurance",
            "Health Plans": "Health Insurance",
            "Group Solutions": "Group Plan",
            "Micro Plans": "Micro Insurance",
            "Combo": "Combo Plan"
        }
        
        standard_category = category_mapping.get(category, category)

        # Generate plan_id for linking brochure and CIS
        plan_id = self._generate_plan_id(insurer, standard_category, product_name)

        metadata = {
            "source": str(file_path),
            "filename": path.name,
            "product_name": product_name,
            "document_type": doc_type,  # "brochure" or "cis"
            "insurer": insurer,
            "insurance_type": standard_category,
            "plan_id": plan_id,  # Links brochure and CIS together
        }

        # Optional: Merge additional static info if product matches exactly
        if product_name in self.external_metadata:
            ext_data = self.external_metadata[product_name]
            if isinstance(ext_data, dict):
                for k, v in ext_data.items():
                    if k == "category":
                        continue 
                    if isinstance(v, list):
                        metadata[k] = ", ".join(v)
                    else:
                        metadata[k] = v
        
        return metadata

    @staticmethod
    def detect_section(text: str) -> str:
        """
        Detect the most likely section based on content headers.
        Returns the section name or 'General' if no match.
        """
        # Check first 500 chars for section headers
        sample = text[:500].lower()
        
        section_scores = {}
        for header in SECTION_HEADERS:
            if header.lower() in sample:
                section_scores[header] = sample.index(header.lower())
        
        if section_scores:
            # Return the earliest matching section
            return min(section_scores, key=section_scores.get)
        
        # Keyword-based fallback detection
        keyword_map = {
            "Eligibility": ["age", "entry age", "minimum age", "maximum age", "who can"],
            "Benefits": ["death benefit", "maturity benefit", "survival benefit", "sum assured"],
            "Exclusions": ["not covered", "excluded", "suicide", "war", "pre-existing"],
            "Charges": ["premium", "fund management", "mortality", "allocation", "admin"],
            "Tax Benefits": ["80c", "80d", "10(10d)", "income tax", "tax benefit"],
            "Riders": ["rider", "accidental", "critical illness", "waiver"],
            "Claim Process": ["claim", "intimation", "documents required", "settlement"]
        }
        
        for section, keywords in keyword_map.items():
            if any(kw in sample for kw in keywords):
                return section
        
        return "General"


# Quick validation block
if __name__ == "__main__":
    extractor = MetadataExtractor("docs")
    
    # Test brochure
    sample1 = "docs/TATA AIA/Term Plans/TATA AIA Smart Value Income Plan Brochure.pdf"
    print("Brochure:", extractor.extract_from_path(sample1))
    
    # Test CIS
    sample2 = "docs/TATA AIA/Term Plans/TATA AIA Maha Raksha Supreme Select CIS.docx"
    print("CIS:", extractor.extract_from_path(sample2))
    
    # Test section detection
    test_text = "Eligibility Criteria: The minimum entry age is 18 years and maximum is 65 years."
    print("Section detected:", MetadataExtractor.detect_section(test_text))
