"""
Input validation and sanitization utilities.
"""
import re
import os
from typing import Optional, Dict, Any
from pathlib import Path
from config import config


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Centralized input validation."""
    
    @staticmethod
    def validate_file_upload(filename: str, file_size_bytes: int) -> bool:
        """
        Validate uploaded file.
        
        Args:
            filename: Name of the uploaded file
            file_size_bytes: Size of the file in bytes
        
        Raises:
            ValidationError: If validation fails
        
        Returns:
            True if valid
        """
        # Check file extension
        ext = Path(filename).suffix.lower().lstrip('.')
        if ext not in config.ALLOWED_FILE_TYPES:
            raise ValidationError(
                f"Invalid file type '.{ext}'. Allowed types: {', '.join(config.ALLOWED_FILE_TYPES)}"
            )
        
        # Check file size
        max_size_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size_bytes > max_size_bytes:
            raise ValidationError(
                f"File size ({file_size_bytes / 1024 / 1024:.2f} MB) exceeds maximum allowed size ({config.MAX_FILE_SIZE_MB} MB)"
            )
        
        if file_size_bytes == 0:
            raise ValidationError("File is empty")
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal.
        
        Args:
            filename: Original filename
        
        Returns:
            Sanitized filename
        """
        # Remove any path components
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        filename = re.sub(r'[^\w\s\-\.]', '_', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure filename is not empty after sanitization
        if not filename:
            filename = "unnamed_file"
        
        return filename
    
    @staticmethod
    def sanitize_path(path: str, base_dir: str) -> str:
        """
        Sanitize and validate file path to prevent directory traversal.
        
        Args:
            path: User-provided path
            base_dir: Base directory that path must be within
        
        Raises:
            ValidationError: If path is outside base directory
        
        Returns:
            Sanitized absolute path
        """
        # Resolve to absolute path
        abs_base = os.path.abspath(base_dir)
        abs_path = os.path.abspath(os.path.join(base_dir, path))
        
        # Check if path is within base directory
        if not abs_path.startswith(abs_base):
            raise ValidationError("Invalid path: directory traversal detected")
        
        return abs_path
    
    @staticmethod
    def validate_query_input(query: str, max_length: int = 10000) -> bool:
        """
        Validate user query input.
        
        Args:
            query: User query string
            max_length: Maximum allowed length
        
        Raises:
            ValidationError: If validation fails
        
        Returns:
            True if valid
        """
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        
        if len(query) > max_length:
            raise ValidationError(f"Query too long (max {max_length} characters)")
        
        # Check for suspicious patterns (basic XSS prevention)
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'onerror=',
            r'onclick=',
        ]
        
        query_lower = query.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, query_lower):
                raise ValidationError("Query contains potentially malicious content")
        
        return True
    
    @staticmethod
    def validate_metadata_filters(filters: Dict[str, Any]) -> bool:
        """
        Validate metadata filters.
        
        Args:
            filters: Filter dictionary
        
        Raises:
            ValidationError: If validation fails
        
        Returns:
            True if valid
        """
        if not isinstance(filters, dict):
            raise ValidationError("Filters must be a dictionary")
        
        # Whitelist of allowed filter keys
        allowed_keys = {
            'insurer', 'insurance_type', 'product_name', 
            'document_type', 'section', 'plan_id'
        }
        
        for key in filters.keys():
            if key not in allowed_keys:
                raise ValidationError(f"Invalid filter key: {key}")
        
        # Validate filter values
        for key, value in filters.items():
            if isinstance(value, str):
                if len(value) > 500:
                    raise ValidationError(f"Filter value too long for key: {key}")
            elif isinstance(value, list):
                if len(value) > 50:
                    raise ValidationError(f"Too many values in filter list for key: {key}")
                for item in value:
                    if isinstance(item, str) and len(item) > 500:
                        raise ValidationError(f"Filter value too long in list for key: {key}")
        
        return True
    
    @staticmethod
    def validate_calculation_inputs(
        age: Optional[int] = None,
        premium_amount: Optional[float] = None,
        policy_term: Optional[str] = None,
        payment_term: Optional[str] = None
    ) -> bool:
        """
        Validate inputs for benefit calculations.
        
        Raises:
            ValidationError: If validation fails
        
        Returns:
            True if valid
        """
        if age is not None:
            if not isinstance(age, int) or age < 0 or age > 120:
                raise ValidationError(f"Invalid age: {age}. Age must be between 0 and 120")
        
        if premium_amount is not None:
            if not isinstance(premium_amount, (int, float)) or premium_amount <= 0:
                raise ValidationError(f"Invalid premium amount: {premium_amount}. Must be positive")
            
            # Reasonable bounds (1000 to 1 crore)
            if premium_amount < 1000 or premium_amount > 10000000:
                raise ValidationError(
                    f"Premium amount {premium_amount} outside reasonable range (₹1,000 - ₹1,00,00,000)"
                )
        
        if policy_term is not None:
            # Extract number from policy term
            pt_match = re.search(r'\d+', str(policy_term))
            if pt_match:
                pt_years = int(pt_match.group())
                if pt_years < 1 or pt_years > 100:
                    raise ValidationError(f"Invalid policy term: {pt_years} years. Must be between 1 and 100")
        
        if payment_term is not None:
            # Extract number from payment term
            ppt_match = re.search(r'\d+', str(payment_term))
            if ppt_match:
                ppt_years = int(ppt_match.group())
                if ppt_years < 1 or ppt_years > 100:
                    raise ValidationError(f"Invalid payment term: {ppt_years} years. Must be between 1 and 100")
        
        return True
    
    @staticmethod
    def validate_api_key(provided_key: Optional[str]) -> bool:
        """
        Validate API key if authentication is enabled.
        
        Args:
            provided_key: API key provided by client
        
        Raises:
            ValidationError: If validation fails
        
        Returns:
            True if valid or auth disabled
        """
        if not config.ENABLE_API_KEY_AUTH:
            return True
        
        if not provided_key:
            raise ValidationError("API key required but not provided")
        
        if provided_key != config.API_KEY:
            raise ValidationError("Invalid API key")
        
        return True
