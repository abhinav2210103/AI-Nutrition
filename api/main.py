from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from string import Template

# These agents/tools are optional and may not be available in every environment;
# import them safely and provide fallbacks to avoid import-time crashes.
try:
    from langchain_tavily import TavilySearch
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent
except Exception:
    TavilySearch = None
    ChatGoogleGenerativeAI = None
    create_react_agent = None

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# ---------- Pydantic Models ----------

class NutritionInfo(BaseModel):
    calories: str
    protein: str
    carbs: str
    fat: str


class BudgetMeal(BaseModel):
    meal_name: str
    ingredients: List[str]
    cooking_steps: List[str]
    nutrition_info: NutritionInfo
    reason_for_selection: str


class BudgetBasedMeal(BaseModel):
    low_budget: BudgetMeal
    medium_budget: BudgetMeal
    high_budget: BudgetMeal


class ThreeMealPlan(BaseModel):
    meal_type: str
    breakfast: BudgetBasedMeal
    lunch: BudgetBasedMeal
    dinner: BudgetBasedMeal


class FourMealPlan(BaseModel):
    meal_type: str
    breakfast: BudgetBasedMeal
    lunch: BudgetBasedMeal
    snack: BudgetBasedMeal
    dinner: BudgetBasedMeal


class IntermittentFastingPlan(BaseModel):
    meal_type: str
    lunch: BudgetBasedMeal
    dinner: BudgetBasedMeal


# 🚨 NEW: full request body includes personal details + preferences
class MealPlanRequest(BaseModel):
    age: int
    height: str
    weight: str
    gender: str
    dietary_preference: List[str]
    allergies: List[str]
    meal_plan_type: str = "3 meals/day"  # "3 meals/day", "4 meals/day", "Intermittent fasting (2 meals)"
    package: str = ""                   # goal / package name
    day: str = "today"                  # e.g., "today", "tomorrow"


# ---------- Prompt Template ----------

prompt_template = Template(
    """
You are a top nutritionist specializing in personalized meal planning. Based on the user's profile, create a personalized meal plan for $day that STRICTLY adheres to their dietary preferences and COMPLETELY EXCLUDES any ingredients they are allergic to. The user follows the "$meal_plan_type" pattern.

IMPORTANT DIETARY GUIDELINES:
1. STRICTLY follow the user's dietary preference: $dietary_preference
2. ABSOLUTELY AVOID any ingredients listed in allergies/restrictions: $allergies
   - Double-check each ingredient to ensure NO allergens are included
   - If an ingredient could contain hidden allergens, choose a safe alternative

Meal Plan Type Guide:
- "3 meals/day" → breakfast, lunch, dinner
- "4 meals/day" → breakfast, lunch, snack, dinner
- "Intermittent fasting (2 meals)" → lunch, dinner

Each meal should have **3 flexible options** while maintaining dietary requirements:
- low budget (affordable while meeting dietary needs)
- medium budget (balanced options within dietary restrictions)
- high budget (premium ingredients following dietary preferences)

Each option must include:
- Meal name (clearly indicating it follows dietary preferences)
- Ingredients (all safe and compliant with dietary restrictions)
- Cooking steps
- Basic nutrition info (calories, protein, carbs, fat)
- Short reason for choosing this meal based on the user's chosen package and dietary needs

User Profile:
- Age: $age
- Height: $height
- Weight: $weight
- Gender: $gender
- Dietary preference: $dietary_preference (STRICT ADHERENCE REQUIRED)
- Allergies or restrictions: $allergies (MUST BE COMPLETELY AVOIDED)
- Goal/package: $package

Output format must be clean and JSON-like, without extra keys. Just the meals as per plan type with 3 budget-based options each. Every meal MUST comply with dietary preferences and exclude allergens.
"""
)

# ---------- LLM + Tools Setup ----------

if TavilySearch and ChatGoogleGenerativeAI and create_react_agent and google_api_key and tavily_api_key:
    tavily_tool = TavilySearch(
        max_results=20,
        topic="general",
        include_answer=True,
        include_raw_content=True,
        search_depth="advanced",
        tavily_api_key=tavily_api_key,
        include_domains=[
            "https://www.nutritionvalue.org/",
            "https://www.walmart.com/search?q=",
            "https://www.healthline.com/nutrition",
            "https://www.healthline.com/nutrition/meal-kits",
            "https://www.healthline.com/nutrition/meal-kits/diets",
            "https://www.healthline.com/nutrition/special-diets",
            "https://www.healthline.com/nutrition/healthy-eating",
            "https://www.healthline.com/nutrition/food-freedom",
            "https://www.healthline.com/nutrition/feel-good-food",
            "https://www.healthline.com/nutrition/products",
            "https://www.healthline.com/nutrition/vitamins-supplements",
            "https://www.healthline.com/nutrition/sustain",
        ],
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-preview-03-25",
        google_api_key=google_api_key,
        temperature=0.5,
    )

    agent_3_meals = create_react_agent(llm, tools=[tavily_tool], response_format=ThreeMealPlan)
    agent_4_meals = create_react_agent(llm, tools=[tavily_tool], response_format=FourMealPlan)
    agent_intermittent = create_react_agent(llm, tools=[tavily_tool], response_format=IntermittentFastingPlan)
else:
    tavily_tool = None
    llm = None
    agent_3_meals = None
    agent_4_meals = None
    agent_intermittent = None


# ---------- FastAPI App ----------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://testing.d1jeqe50o5hnb3.amplifyapp.com",
        "https://www.dailywellness.ai",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Nutrition Meal Plan API"}


# ---------- Core Helper (NO DB, NO USER ID) ----------

async def generate_meal_plan_for_profile(profile: MealPlanRequest):
    """
    Generate a meal plan directly from the input profile.
    No database, no user_id. Everything comes from the request body.
    """
    try:
        if not (agent_3_meals and agent_4_meals and agent_intermittent):
            raise HTTPException(
                status_code=500,
                detail="LLM agents are not properly configured (missing keys or libraries or API keys).",
            )

        prompt_data = {
            "age": profile.age,
            "height": profile.height,
            "weight": profile.weight,
            "gender": profile.gender,
            "dietary_preference": ", ".join(profile.dietary_preference),
            "allergies": ", ".join(profile.allergies),
            "package": profile.package,
            "meal_plan_type": profile.meal_plan_type,
            "day": profile.day,
        }

        filled_prompt = prompt_template.substitute(**prompt_data)
        inputs = {"messages": [("user", filled_prompt)]}

        # Choose agent based on meal_plan_type
        if profile.meal_plan_type.startswith("3 meals"):
            result = await agent_3_meals.ainvoke(inputs)
            meal_plan = result["structured_response"]
        elif profile.meal_plan_type.startswith("4 meals"):
            result = await agent_4_meals.ainvoke(inputs)
            meal_plan = result["structured_response"]
        else:
            result = await agent_intermittent.ainvoke(inputs)
            meal_plan = result["structured_response"]

        return meal_plan
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating meal plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Public APIs (Stateless) ----------

@app.post("/v1/generate-meal-plan")
async def generate_meal_plan(request: MealPlanRequest):
    """
    Generate a meal plan based purely on the request body.
    No auth, no user_id, no MongoDB – completely stateless.
    """
    return await generate_meal_plan_for_profile(request)


@app.post("/nutrition-images")
def get_image_urls(meal_plan_type: Optional[str] = None):
    all_meals = {
        "breakfast": {
            "low_budget": "https://res.cloudinary.com/dsjswlbmx/image/upload/v1745129565/images/breakfast/c93ejgdtcpfqvgqid1th.jpg",
            "medium_budget": "https://res.cloudinary.com/dsjswlbmx/image/upload/v1745129565/images/breakfast/h1lcqnyrhpxt5s4tx9xm.jpg",
            "high_budget": "https://res.cloudinary.com/dsjswlbmx/image/upload/v1745129565/images/breakfast/fdgg4yp3sgazvjfcjtzo.jpg",
        },
        "lunch": {
            "low_budget": "https://res.cloudinary.com/dsjswlbmx/image/upload/v1745129565/images/lunch/zldpuvakmzl02oqy5ouw.jpg",
            "medium_budget": "https://res.cloudinary.com/dsjswlbmx/image/upload/v1745129565/images/lunch/a9vntmgpcoglwlmggh5j.jpg",
            "high_budget": "https://res.cloudinary.com/dsjswlbmx/image/upload/v1745129565/images/lunch/amfrudbzdqgquk6jdpil.jpg",
        },
        "snack": {
            "low_budget": "https://res.cloudinary.com/dsjswlbmx/image/upload/v1745129566/images/snack/r0d3dikwiwslxfkkk9ny.jpg",
            "medium_budget": "https://res.cloudinary.com/dsjswlbmx/image/upload/v1745129566/images/snack/dlh1dcscg17w2qb68pat.jpg",
            "high_budget": "https://res.cloudinary.com/dsjswlbmx/image/upload/v1745129565/images/snack/um0accdew1ikmvda4s30.jpg",
        },
        "dinner": {
            "low_budget": "https://res.cloudinary.com/dsjswlbmx/image/upload/v1745129565/images/dinner/niijz8nu9ojtvpchftwx.jpg",
            "medium_budget": "https://res.cloudinary.com/dsjswlbmx/image/upload/v1745129565/images/dinner/zphqbwhualjza9cmxswz.jpg",
            "high_budget": "https://res.cloudinary.com/dsjswlbmx/image/upload/v1745129565/images/dinner/kavnsjtvdzqoicxam29y.jpg",
        },
    }

    if meal_plan_type and meal_plan_type.startswith("3 meals"):
        return {
            "breakfast": all_meals["breakfast"],
            "lunch": all_meals["lunch"],
            "dinner": all_meals["dinner"],
        }
    elif meal_plan_type and meal_plan_type.startswith("4 meals"):
        return all_meals
    else:
        return {
            "lunch": all_meals["lunch"],
            "dinner": all_meals["dinner"],
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
