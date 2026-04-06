"""
AI Scoring Engine for Location Guide App.

Calculates multi-factor scores for places based on rating, sentiment, crowd level,
user preferences, time of day, distance, and price. Includes category-specific
scoring logic for restaurants, beaches, temples, and malls.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

SCORING_WEIGHTS = {
    "restaurants": {
        "rating": 0.15,
        "sentiment": 0.15,
        "food_quality": 0.20,
        "service": 0.10,
        "price_value": 0.10,
        "ambiance": 0.10,
        "time_fit": 0.10,
        "preference_match": 0.10,
    },
    "beaches": {
        "rating": 0.15,
        "sentiment": 0.10,
        "cleanliness": 0.20,
        "crowd_level": 0.15,
        "activities": 0.10,
        "time_fit": 0.15,
        "preference_match": 0.15,
    },
    "temples": {
        "rating": 0.15,
        "sentiment": 0.10,
        "significance": 0.20,
        "ritual_availability": 0.10,
        "crowd_level": 0.10,
        "time_fit": 0.15,
        "preference_match": 0.20,
    },
    "malls": {
        "rating": 0.10,
        "sentiment": 0.10,
        "store_variety": 0.15,
        "parking": 0.10,
        "food_court": 0.10,
        "events": 0.10,
        "time_fit": 0.15,
        "preference_match": 0.20,
    },
    "default": {
        "rating": 0.20,
        "sentiment": 0.15,
        "crowd_level": 0.10,
        "time_fit": 0.15,
        "preference_match": 0.20,
        "distance": 0.10,
        "price_value": 0.10,
    },
}

TIME_SUITABILITY = {
    "restaurants": {
        "morning": {"breakfast": 1.0, "cafe": 0.9, "bakery": 0.8},
        "afternoon": {"lunch": 1.0, "casual_dining": 0.8, "fast_food": 0.9},
        "evening": {"dinner": 1.0, "fine_dining": 1.0, "bar": 0.9},
        "night": {"late_night": 0.9, "bar": 1.0, "club": 0.8},
    },
    "beaches": {
        "morning": {"sunrise": 1.0, "swimming": 0.8, "jogging": 0.9},
        "afternoon": {"swimming": 0.7, "sunbathing": 0.8, "water_sports": 0.9},
        "evening": {"sunset": 1.0, "walk": 0.9, "photography": 0.8},
        "night": {"bonfire": 0.7, "night_walk": 0.5},
    },
    "temples": {
        "morning": {"puja": 1.0, "meditation": 0.9, "darshan": 1.0},
        "afternoon": {"darshan": 0.7, "tour": 0.6},
        "evening": {"aarti": 1.0, "darshan": 0.9, "meditation": 0.8},
        "night": {"night_prayer": 0.6},
    },
    "malls": {
        "morning": {"shopping": 0.6, "grocery": 0.7},
        "afternoon": {"shopping": 0.9, "food_court": 0.8, "movies": 0.7},
        "evening": {"shopping": 1.0, "dining": 0.9, "entertainment": 1.0},
        "night": {"movies": 0.8, "late_shopping": 0.5},
    },
}

CROWD_PENALTY_BY_HOUR = {
    "restaurants": {
        12: 0.15, 13: 0.15, 19: 0.20, 20: 0.20, 21: 0.15,
    },
    "beaches": {
        11: 0.10, 12: 0.15, 13: 0.15, 14: 0.10, 15: 0.10,
    },
    "temples": {
        6: 0.10, 7: 0.15, 8: 0.15, 18: 0.15, 19: 0.10,
    },
    "malls": {
        11: 0.10, 12: 0.10, 15: 0.10, 16: 0.15, 17: 0.15, 18: 0.20, 19: 0.20, 20: 0.15,
    },
}


def calculate_ai_score(
    place: dict,
    user_prefs: dict | None = None,
    time_of_day: str | None = None,
    current_hour: int | None = None,
) -> dict[str, Any]:
    """
    Calculate a comprehensive AI score for a place.

    Args:
        place: Place data dict with rating, reviews, category, and category-specific fields.
        user_prefs: User preferences (budget, vibe, dietary, interests, etc.).
        time_of_day: Context (morning/afternoon/evening/night).
        current_hour: Current hour 0-23.

    Returns:
        Dict with total_score (0-100), component_scores, and explanation.
    """
    user_prefs = user_prefs or {}
    category = place.get("category", "default").lower()
    weights = SCORING_WEIGHTS.get(category, SCORING_WEIGHTS["default"])

    rating_score = _score_rating(place)
    sentiment_score = _score_sentiment(place)
    crowd_score = _score_crowd(place, current_hour)
    time_score = _score_time_fit(category, time_of_day, current_hour, place)
    pref_score = _score_preference_match(place, user_prefs)
    distance_score = _score_distance(place, user_prefs)
    price_score = _score_price_value(place, user_prefs)

    category_scores = _score_category_specific(category, place, time_of_day, current_hour)

    components = {
        "rating": rating_score,
        "sentiment": sentiment_score,
        "crowd_level": crowd_score,
        "time_fit": time_score,
        "preference_match": pref_score,
        "distance": distance_score,
        "price_value": price_score,
    }
    components.update(category_scores)

    total = 0.0
    for component_name, score in components.items():
        weight = weights.get(component_name, 0.0)
        total += score * weight

    total = min(max(total * 100, 0), 100)

    explanation = _generate_explanation(category, components, weights, total)

    return {
        "total_score": round(total, 2),
        "component_scores": {k: round(v, 3) for k, v in components.items()},
        "weights_used": weights,
        "explanation": explanation,
    }


def _score_rating(place: dict) -> float:
    """Score based on place rating, normalized to 0-1."""
    rating = place.get("rating", place.get("avg_rating", 3.0))
    try:
        rating = float(rating)
    except (ValueError, TypeError):
        rating = 3.0

    review_count = place.get("review_count", place.get("num_reviews", 0))
    try:
        review_count = int(review_count)
    except (ValueError, TypeError):
        review_count = 0

    normalized = rating / 5.0

    confidence = min(review_count / 50.0, 1.0)
    adjusted = normalized * (0.5 + 0.5 * confidence)

    return max(min(adjusted, 1.0), 0.0)


def _score_sentiment(place: dict) -> float:
    """Score based on aggregate sentiment from reviews."""
    sentiment = place.get("sentiment_score", place.get("aggregate_sentiment", None))

    if sentiment is not None:
        try:
            sentiment = float(sentiment)
            return max(min((sentiment + 1.0) / 2.0, 1.0), 0.0)
        except (ValueError, TypeError):
            pass

    reviews = place.get("reviews", [])
    if not reviews:
        return 0.5

    positive = 0
    total = 0
    for review in reviews:
        if isinstance(review, dict):
            rating = review.get("rating", review.get("score", 3))
            try:
                rating = float(rating)
            except (ValueError, TypeError):
                rating = 3.0
            if rating >= 3.5:
                positive += 1
            total += 1
        elif isinstance(review, str) and len(review) > 10:
            total += 1
            positive += 0.6

    return positive / max(total, 1)


def _score_crowd(place: dict, current_hour: int | None = None) -> float:
    """Score based on crowd level (lower crowd = higher score for most contexts)."""
    crowd_level = place.get("crowd_level", place.get("busyness", "medium"))
    category = place.get("category", "default").lower()

    crowd_map = {"low": 0.9, "moderate": 0.7, "medium": 0.6, "high": 0.3, "very_high": 0.15}
    base_score = crowd_map.get(str(crowd_level).lower(), 0.5)

    if current_hour is not None:
        hourly_penalty = CROWD_PENALTY_BY_HOUR.get(category, {})
        penalty = hourly_penalty.get(current_hour, 0.0)
        base_score *= (1.0 - penalty)

    return max(min(base_score, 1.0), 0.0)


def _score_time_fit(
    category: str,
    time_of_day: str | None,
    current_hour: int | None,
    place: dict,
) -> float:
    """Score how well the place fits the current time context."""
    if not time_of_day and current_hour is None:
        return 0.5

    if current_hour is not None and time_of_day is None:
        if 5 <= current_hour < 11:
            time_of_day = "morning"
        elif 11 <= current_hour < 16:
            time_of_day = "afternoon"
        elif 16 <= current_hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

    time_suit = TIME_SUITABILITY.get(category, {})
    time_scores = time_suit.get(time_of_day or "", {})

    if time_scores:
        place_type = place.get("type", place.get("subcategory", place.get("cuisine", "")))
        if isinstance(place_type, str):
            place_type = place_type.lower()
            for key, score in time_scores.items():
                if key in place_type or place_type in key:
                    return score
        return sum(time_scores.values()) / len(time_scores)

    open_hours = place.get("open_hours", place.get("operating_hours", {}))
    if isinstance(open_hours, dict) and current_hour is not None:
        opens = open_hours.get("open", open_hours.get("opens", 0))
        closes = open_hours.get("close", open_hours.get("closes", 23))
        try:
            opens = int(opens)
            closes = int(closes)
        except (ValueError, TypeError):
            return 0.5

        if opens <= current_hour <= closes:
            mid = (opens + closes) / 2
            distance_from_mid = abs(current_hour - mid) / max((closes - opens) / 2, 1)
            return max(1.0 - distance_from_mid * 0.3, 0.5)
        return 0.1

    return 0.5


def _score_preference_match(place: dict, user_prefs: dict) -> float:
    """Score how well the place matches user preferences."""
    if not user_prefs:
        return 0.5

    scores = []

    budget = user_prefs.get("budget", user_prefs.get("price_preference"))
    if budget:
        place_price = place.get("price_range", place.get("price_level", ""))
        price_map = {"low": 1, "budget": 1, "moderate": 2, "medium": 2, "high": 3, "expensive": 3, "luxury": 4}
        place_level = price_map.get(str(place_price).lower(), 2)
        pref_level = price_map.get(str(budget).lower(), 2)
        budget_score = 1.0 - min(abs(place_level - pref_level) / 3.0, 1.0)
        scores.append(budget_score)

    vibe = user_prefs.get("vibe", user_prefs.get("atmosphere"))
    if vibe:
        place_vibes = place.get("vibes", place.get("atmosphere", place.get("tags", [])))
        if isinstance(place_vibes, str):
            place_vibes = [place_vibes]
        if isinstance(place_vibes, list):
            vibe_lower = str(vibe).lower()
            vibe_match = any(vibe_lower in str(pv).lower() for pv in place_vibes)
            scores.append(1.0 if vibe_match else 0.3)

    dietary = user_prefs.get("dietary", user_prefs.get("diet"))
    if dietary:
        options = place.get("dietary_options", place.get("cuisine_types", place.get("tags", [])))
        if isinstance(options, str):
            options = [options]
        if isinstance(options, list):
            dietary_lower = str(dietary).lower()
            dietary_match = any(dietary_lower in str(o).lower() for o in options)
            scores.append(1.0 if dietary_match else 0.4)

    interests = user_prefs.get("interests", [])
    if interests:
        if isinstance(interests, str):
            interests = [interests]
        place_features = place.get("features", place.get("amenities", place.get("highlights", [])))
        if isinstance(place_features, str):
            place_features = [place_features]
        if isinstance(place_features, list):
            features_lower = [str(f).lower() for f in place_features]
            matches = sum(1 for i in interests if any(str(i).lower() in f for f in features_lower))
            scores.append(min(matches / max(len(interests), 1), 1.0))

    if not scores:
        return 0.5

    return sum(scores) / len(scores)


def _score_distance(place: dict, user_prefs: dict) -> float:
    """Score based on distance from user (closer = better)."""
    distance = user_prefs.get("distance_km", place.get("distance_km", place.get("distance", None)))

    if distance is None:
        return 0.5

    try:
        distance = float(distance)
    except (ValueError, TypeError):
        return 0.5

    if distance <= 1:
        return 1.0
    elif distance <= 5:
        return 0.8
    elif distance <= 10:
        return 0.6
    elif distance <= 25:
        return 0.4
    else:
        return 0.2


def _score_price_value(place: dict, user_prefs: dict) -> float:
    """Score based on price-to-value ratio."""
    rating = place.get("rating", 3.0)
    try:
        rating = float(rating)
    except (ValueError, TypeError):
        rating = 3.0

    price_level = place.get("price_range", place.get("price_level", "moderate"))
    price_map = {"low": 1, "budget": 1, "$": 1, "moderate": 2, "medium": 2, "$$": 2, "high": 3, "expensive": 3, "$$$": 3, "luxury": 4, "$$$$": 4}
    price_num = price_map.get(str(price_level).lower(), 2)

    value_ratio = (rating / 5.0) / max(price_num / 4.0, 0.25)
    return max(min(value_ratio, 1.0), 0.0)


def _score_category_specific(
    category: str,
    place: dict,
    time_of_day: str | None,
    current_hour: int | None,
) -> dict[str, float]:
    """Calculate category-specific scores."""
    if category == "restaurants":
        return _score_restaurants(place)
    elif category == "beaches":
        return _score_beaches(place, time_of_day, current_hour)
    elif category == "temples":
        return _score_temples(place, time_of_day, current_hour)
    elif category == "malls":
        return _score_malls(place, time_of_day, current_hour)
    return {}


def _score_restaurants(place: dict) -> dict[str, float]:
    """Score restaurants on food quality, service, price, and ambiance."""
    food_quality = place.get("food_quality", place.get("food_rating", None))
    if food_quality is not None:
        try:
            food_quality = min(float(food_quality) / 5.0, 1.0)
        except (ValueError, TypeError):
            food_quality = 0.5
    else:
        food_quality = 0.5

    service = place.get("service_rating", place.get("service", None))
    if service is not None:
        try:
            service = min(float(service) / 5.0, 1.0)
        except (ValueError, TypeError):
            service = 0.5
    else:
        service = 0.5

    price = place.get("price_range", place.get("price_level", "moderate"))
    price_map = {"low": 1.0, "budget": 0.9, "moderate": 0.7, "medium": 0.7, "high": 0.5, "expensive": 0.4, "luxury": 0.3}
    price_score = price_map.get(str(price).lower(), 0.5)

    ambiance = place.get("ambiance", place.get("ambiance_rating", place.get("atmosphere_rating", None)))
    if ambiance is not None:
        try:
            ambiance = min(float(ambiance) / 5.0, 1.0)
        except (ValueError, TypeError):
            ambiance = 0.5
    else:
        ambiance = 0.5

    return {
        "food_quality": food_quality,
        "service": service,
        "price_value": price_score,
        "ambiance": ambiance,
    }


def _score_beaches(
    place: dict,
    time_of_day: str | None,
    current_hour: int | None,
) -> dict[str, float]:
    """Score beaches on cleanliness, crowd, activities, and best time."""
    cleanliness = place.get("cleanliness", place.get("cleanliness_rating", None))
    if cleanliness is not None:
        try:
            cleanliness = min(float(cleanliness) / 5.0, 1.0)
        except (ValueError, TypeError):
            cleanliness = 0.5
    else:
        cleanliness = 0.5

    crowd = place.get("crowd_level", "medium")
    crowd_map = {"low": 0.9, "moderate": 0.7, "medium": 0.6, "high": 0.3, "very_high": 0.15}
    crowd_score = crowd_map.get(str(crowd).lower(), 0.5)

    activities = place.get("activities", place.get("available_activities", []))
    if isinstance(activities, list):
        activity_score = min(len(activities) / 5.0, 1.0)
    elif isinstance(activities, str):
        activity_score = 0.6 if activities else 0.3
    else:
        activity_score = 0.5

    best_time = place.get("best_time", place.get("ideal_time", ""))
    time_score = 0.5
    if best_time and time_of_day:
        if str(best_time).lower() == str(time_of_day).lower():
            time_score = 1.0
        elif str(time_of_day).lower() in str(best_time).lower():
            time_score = 0.8

    return {
        "cleanliness": cleanliness,
        "crowd_level": crowd_score,
        "activities": activity_score,
        "time_fit": time_score,
    }


def _score_temples(
    place: dict,
    time_of_day: str | None,
    current_hour: int | None,
) -> dict[str, float]:
    """Score temples on rituals, dress code, significance, and peak hours."""
    significance = place.get("significance", place.get("historical_significance", place.get("importance", None)))
    if significance is not None:
        try:
            significance = min(float(significance) / 5.0, 1.0)
        except (ValueError, TypeError):
            significance = 0.5
    else:
        significance = 0.5

    rituals = place.get("rituals", place.get("ritual_availability", place.get("puja_available", None)))
    if rituals is not None:
        if isinstance(rituals, bool):
            ritual_score = 0.8 if rituals else 0.2
        elif isinstance(rituals, list):
            ritual_score = min(len(rituals) / 5.0, 1.0)
        else:
            ritual_score = 0.6
    else:
        ritual_score = 0.5

    dress_code = place.get("dress_code", place.get("dress_code_required", None))
    dress_score = 0.7
    if dress_code is not None:
        dress_score = 0.8

    crowd = place.get("crowd_level", "medium")
    crowd_map = {"low": 0.9, "moderate": 0.7, "medium": 0.6, "high": 0.4, "very_high": 0.2}
    crowd_score = crowd_map.get(str(crowd).lower(), 0.5)

    peak_hours = place.get("peak_hours", place.get("busy_hours", []))
    time_score = 0.7
    if peak_hours and current_hour is not None:
        try:
            if isinstance(peak_hours, list) and len(peak_hours) >= 2:
                peak_start = int(peak_hours[0])
                peak_end = int(peak_hours[1])
                if peak_start <= current_hour <= peak_end:
                    time_score = 0.4
        except (ValueError, TypeError, IndexError):
            pass

    return {
        "significance": significance,
        "ritual_availability": ritual_score,
        "crowd_level": crowd_score,
        "time_fit": time_score,
    }


def _score_malls(
    place: dict,
    time_of_day: str | None,
    current_hour: int | None,
) -> dict[str, float]:
    """Score malls on stores, parking, food court, and events."""
    stores = place.get("store_count", place.get("num_stores", place.get("stores", None)))
    if stores is not None:
        try:
            stores = int(stores)
            store_score = min(stores / 200.0, 1.0)
        except (ValueError, TypeError):
            store_score = 0.5
    else:
        store_score = 0.5

    parking = place.get("parking", place.get("parking_available", place.get("parking_capacity", None)))
    if parking is not None:
        if isinstance(parking, bool):
            parking_score = 0.7 if parking else 0.2
        elif isinstance(parking, str):
            parking_map = {"ample": 0.9, "good": 0.8, "limited": 0.4, "none": 0.1}
            parking_score = parking_map.get(parking.lower(), 0.5)
        else:
            try:
                parking_score = min(float(parking) / 1000.0, 1.0)
            except (ValueError, TypeError):
                parking_score = 0.5
    else:
        parking_score = 0.5

    food_court = place.get("food_court", place.get("food_court_available", place.get("dining_options", None)))
    if food_court is not None:
        if isinstance(food_court, bool):
            food_score = 0.7 if food_court else 0.2
        elif isinstance(food_court, int):
            food_score = min(food_court / 20.0, 1.0)
        else:
            food_score = 0.6
    else:
        food_score = 0.5

    events = place.get("events", place.get("events_available", place.get("upcoming_events", [])))
    if isinstance(events, list):
        event_score = min(len(events) / 5.0, 1.0)
    elif isinstance(events, bool):
        event_score = 0.6 if events else 0.3
    else:
        event_score = 0.4

    return {
        "store_variety": store_score,
        "parking": parking_score,
        "food_court": food_score,
        "events": event_score,
    }


def _generate_explanation(
    category: str,
    components: dict[str, float],
    weights: dict[str, float],
    total: float,
) -> str:
    """Generate a human-readable explanation of the score."""
    top_factors = sorted(
        [(name, score, weights.get(name, 0)) for name, score in components.items()],
        key=lambda x: x[1] * x[2],
        reverse=True,
    )[:3]

    factor_strs = []
    for name, score, weight in top_factors:
        label = name.replace("_", " ")
        if score >= 0.7:
            factor_strs.append(f"strong {label}")
        elif score >= 0.4:
            factor_strs.append(f"moderate {label}")
        else:
            factor_strs.append(f"weak {label}")

    if total >= 75:
        verdict = "Highly recommended"
    elif total >= 55:
        verdict = "Good option"
    elif total >= 35:
        verdict = "Consider with caution"
    else:
        verdict = "Not ideal right now"

    return f"{verdict}. Key factors: {', '.join(factor_strs)}."
