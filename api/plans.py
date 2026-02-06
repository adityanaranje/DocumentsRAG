from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import re
from random import random

router = APIRouter(prefix="/plans", tags=["Plans"])

class PlanInfoRequest(BaseModel):
    age: int
    gender: str
    premium_amount: float
    goal: Optional[str] = "savings"
    plan_id: Optional[int] = None
    policy_term: Optional[str] = None
    payment_term: Optional[str] = None
    payment_mode: Optional[str] = None

class BenefitItem(BaseModel):
    name: str
    value: str
    description: str

class PlanBenefitResponse(BaseModel):
    plan_id: int
    plan_name: str
    eligibility_status: bool
    reason: str
    maturity_benefit: str
    annual_income: str
    sum_assured: str
    income_start_point: str
    income_duration: str
    sad_multiple: str
    payout_freq: str
    recommendation_score: float
    benefits: List[BenefitItem]

# Dummy Plan Data Store
PLANS_DATA = {
    1: {
        "name": "Edelweiss Life Guaranteed Income STAR",
        "min_age": 3,
        "max_age": 50,
        "benefits_multiplier": 1.2,
        "income_start": "5 years",
        "income_duration": "20 years",
        "payout_freq": "Yearly",
        "sad_multiple": "10"
    },
    2: {
        "name": "Edelweiss Life Bharat Savings STAR",
        "min_age": 0,
        "max_age": 60,
        "benefits_multiplier": 1.1,
        "income_start": "2nd year",
        "income_duration": "15 years",
        "payout_freq": "Monthly",
        "sad_multiple": "7"
    },
    3: {
        "name": "Edelweiss Life Premier Guaranteed STAR Pro",
        "min_age": 5,
        "max_age": 55,
        "benefits_multiplier": 1.3,
        "income_start": "15 years",
        "income_duration": "20 years",
        "payout_freq": "Yearly",
        "sad_multiple": "11"
    },
     4: {
        "name": "EdelweissLife Flexi Dream Plan",
        "min_age": 18,
        "max_age": 60,
        "benefits_multiplier": 0.9,
        "income_start": "2 years",
        "income_duration": "10 years",
        "payout_freq": "Yearly",
        "sad_multiple": "8"
    },
    5: {
        "name": "EdelweissLife Guaranteed Savings STAR",
        "min_age": 0,
        "max_age": 60,
        "benefits_multiplier": 1.17,
        "income_start": "3rd year",
        "income_duration": "15 years",
        "payout_freq": "Monthly",
        "sad_multiple": "7"
    },
    6: {
        "name": "EdelweissLife Flexi Savings STAR",
        "min_age": 18,
        "max_age": 65,
        "benefits_multiplier": 1.42,
        "income_start": "10 years",
        "income_duration": "25 years",
        "payout_freq": "Yearly",
        "sad_multiple": "5"
    }
}

# Plan Name to ID Mapping
PLAN_NAME_TO_ID = {
    "guaranteed income star": 1,
    "bharat savings star": 2,
    "premier guaranteed star pro": 3,
    "Flexi Dream Plan": 4,
    "Flexi Savings STAR": 6,
    "Guaranteed Savings STAR": 5
}

def resolve_plan_id(name: str) -> Optional[int]:
    """Resolves a plan name or substring to a Plan ID."""
    name_lower = name.lower().strip()
    for key, pid in PLAN_NAME_TO_ID.items():
        if key in name_lower or name_lower in key:
            return pid
    return None

def calculate_dummy_benefits(plan_id: int, request: PlanInfoRequest) -> PlanBenefitResponse:
    plan = PLANS_DATA.get(plan_id)
    if not plan:
        return None
    
    is_eligible = plan["min_age"] <= request.age <= plan["max_age"]
    reason = "Eligible based on age criteria." if is_eligible else f"Ineligible: Age must be between {plan['min_age']} and {plan['max_age']}"
    
    # Dummy calculation logic influenced by PT/PPT
    pt_val = 15
    if request.policy_term:
        try: pt_val = int(re.search(r'\d+', request.policy_term).group())
        except: pass
    
    mult_adj = (pt_val / 15.0) # PT adjustment
    randval = random()
    if randval < 0.5:
        randval = 0.5
    maturity_val = request.premium_amount * 10 * (1+randval)* plan["benefits_multiplier"] * mult_adj
    income_val = request.premium_amount * (plan["benefits_multiplier"]/0.467)
    
    # Calculate Sum Assured
    sad_val = request.premium_amount * float(plan["sad_multiple"])
    sum_assured = f"₹{sad_val:,.2f}"
    
    maturity_benefit = f"₹{maturity_val:,.2f}"
    annual_income = f"₹{income_val:,.2f}"
    
    benefits = [
        BenefitItem(name="Maturity Benefit", value=maturity_benefit, description="Guaranteed lump sum"),
        BenefitItem(name="Annual Income Benefit", value=annual_income, description="Regular payouts"),
        BenefitItem(name="Sum Assured", value=sum_assured, description="Life Cover"),
        BenefitItem(name="Tax Benefit", value="Exempt", description="Sec 80C")
    ]
    
    return PlanBenefitResponse(
        plan_id=plan_id,
        plan_name=plan["name"],
        eligibility_status=is_eligible,
        reason=reason,
        maturity_benefit=maturity_benefit,
        annual_income=annual_income,
        sum_assured=sum_assured,
        income_start_point=plan["income_start"],
        income_duration=plan["income_duration"],
        payout_freq=plan["payout_freq"],
        sad_multiple=plan["sad_multiple"],
        recommendation_score=0.9 if is_eligible else 0.1,
        benefits=benefits
    )

@router.get("/calculate", response_model=PlanBenefitResponse)
async def calculate_by_id(
    plan_id: int = Query(..., alias="Planid"),
    age: int = Query(...),
    gender: str = Query(...),
    premium_amount: float = Query(...),
    goal: str = Query("savings")
):
    """
    Calculates benefits for a specific Edelweiss plan using Plan ID.
    Uses dummy logic for demonstration.
    """
    request = PlanInfoRequest(age=age, gender=gender, premium_amount=premium_amount, goal=goal)
    result = calculate_dummy_benefits(plan_id, request)
    if not result:
        raise HTTPException(status_code=404, detail="Plan not found")
    return result

@router.post("/calculate", response_model=List[PlanBenefitResponse])
async def calculate_all_benefits(request: PlanInfoRequest):
    """
    Calculates benefits for all Edelweiss Guaranteed Income plans.
    """
    if request.plan_id:
        result = calculate_dummy_benefits(request.plan_id, request)
        if not result:
            raise HTTPException(status_code=404, detail="Plan not found")
        return [result]
    
    results = []
    for pid in PLANS_DATA:
        results.append(calculate_dummy_benefits(pid, request))
    return results

def get_plan_benefits_tool(age: int, gender: str, premium_amount: float, plan_id: Optional[int] = None,
                          policy_term: Optional[str] = None, payment_term: Optional[str] = None,
                          payment_mode: Optional[str] = None) -> str:
    """
    Python function to be used as a tool by LangGraph.
    Returns a combined string with a Markdown table and JSON.
    """
    request = PlanInfoRequest(
        age=age, 
        gender=gender, 
        premium_amount=premium_amount,
        plan_id=plan_id,
        policy_term=policy_term,
        payment_term=payment_term,
        payment_mode=payment_mode
    )
    data = []
    if plan_id:
        result = calculate_dummy_benefits(plan_id, request)
        if result: data = [result.dict()]
    else:
        for pid in PLANS_DATA:
            data.append(calculate_dummy_benefits(pid, request).dict())
    
    if not data:
        return "No plans found or ineligible."

    # Create a nice Markdown Table for the LLM
    table = "| Plan Name | Income Start | Duration | SAD Multi | Sum Assured | Maturity Benefit | Annual Income |\n"
    table += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for d in data:
        table += f"| {d['plan_name']} | {d['income_start_point']} | {d['income_duration']} | {d['sad_multiple']} | {d['sum_assured']} | {d['maturity_benefit']} | {d['annual_income']} |\n"
    
    output = {
        "summary_table": table,
        "raw_data": data
    }
    return json.dumps(output, indent=2)
