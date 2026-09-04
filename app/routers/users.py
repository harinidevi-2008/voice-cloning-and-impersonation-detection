from typing import List
from fastapi import APIRouter

from app.schemas import UserOut
from app.db import crud

router = APIRouter()


@router.get("/users", response_model=List[UserOut])
async def get_users():
    """
    Returns all enrolled users, for populating the dashboard's dropdown
    (used to pick a claimed_user_id when calling /analyze).
    """
    rows = crud.list_users()
    return [
        UserOut(
            user_id=row["user_id"],
            name=row["name"],
            role=row["role"],
            enrolled_at=row["enrolled_at"],
        )
        for row in rows
    ]
