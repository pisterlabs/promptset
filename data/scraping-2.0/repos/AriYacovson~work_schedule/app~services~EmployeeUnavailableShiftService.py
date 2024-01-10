from app.models import ShiftModel
from app.models.employee_unavailable_shift import EmployeeUnavailableShiftModel
from sqlalchemy.exc import SQLAlchemyError
import logging

from app.utils import openai_util

from app.schemas.employee_unavailable_shift_request import (
    EmployeeUnavailableShiftRequest,
)
from app.schemas.employee_unavailable_shift_response import (
    EmployeeUnavailableShiftResponse,
)

# from app.schemas.EmployeeShiftAssignmentRequest import EmployeeShiftAssignmentRequest
# from app.schemas.EmployeeShiftAssignmentResponse import EmployeeShiftAssignmentResponse

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_employee_unavailable_shifts(db) -> list[EmployeeUnavailableShiftResponse]:
    try:
        return db.query(EmployeeUnavailableShiftModel).all()
    except SQLAlchemyError as e:
        logger.error(f"Failed to fetch all unavailable shifts: {e}")
        return None


def get_employee_unavailable_shift_by_id(
    db, employee_unavailable_shift_id: int
) -> EmployeeUnavailableShiftResponse:
    try:
        return (
            db.query(EmployeeUnavailableShiftModel)
            .filter(EmployeeUnavailableShiftModel.id == employee_unavailable_shift_id)
            .first()
        )
    except SQLAlchemyError as e:
        logger.error(
            f"Failed to fetch unavailable shift with id {employee_unavailable_shift_id}: {e}"
        )
        return None


def create_employee_unavailable_shift(
    db, employee_unavailable_shift: EmployeeUnavailableShiftRequest
) -> EmployeeUnavailableShiftResponse:
    employee_unavailable_shift_model = EmployeeUnavailableShiftModel(
        **employee_unavailable_shift.model_dump()
    )
    try:
        db.add(employee_unavailable_shift_model)
        db.commit()
        db.refresh(employee_unavailable_shift_model)
        return employee_unavailable_shift_model
    except SQLAlchemyError as e:
        logger.error(f"Failed to create unavailable shift: {e}")
        return None


def create_employee_unavailable_shifts_from_text(
        db, employee_id: int, text: str
) -> bool:
    unavailable_shifts = openai_util.extract_unavailable_shifts_from_text(
        employee_id, text
    )
    unavailable_shifts = eval(unavailable_shifts)
    for unavailable_shift in unavailable_shifts:
        unavailable_shift_date = unavailable_shift["date"]
        unavailable_shift_type_id = unavailable_shift["shift_type_id"]

        # Query for the corresponding shift id
        shift = db.query(ShiftModel).filter(ShiftModel.date == unavailable_shift_date, ShiftModel.shift_type_id == unavailable_shift_type_id).first()
        if shift:
            shift_id = shift.id

            # Check if this unavailable shift already exists
            existing_unavailable_shift = db.query(EmployeeUnavailableShiftModel).filter(
                EmployeeUnavailableShiftModel.employee_id == employee_id,
                EmployeeUnavailableShiftModel.shift_id == shift_id
            ).first()

            if not existing_unavailable_shift:
                unavailable_shift_model = EmployeeUnavailableShiftModel(
                    employee_id=employee_id, shift_id=shift_id
                )
                db.add(unavailable_shift_model)
    try:
        db.commit()
        return True
    except SQLAlchemyError as e:
        logger.error(f"Failed to create unavailable shift: {e}")
        return False



    # try:
    #     unavailable_shifts = openai_util.extract_unavailable_shifts_from_text(
    #         employee_id, text
    #     )
    #     unavailable_shifts = eval(unavailable_shifts)
    #     for unavailable_shift in unavailable_shifts:
    #         employee_unavailable_shift_model = EmployeeUnavailableShiftModel(
    #             **unavailable_shift
    #         )
    #         db.add(employee_unavailable_shift_model)
    #         db.commit()
    #         db.refresh(employee_unavailable_shift_model)
    #     return True
    # except SQLAlchemyError as e:
    #     logger.error(f"Failed to create unavailable shift: {e}")
    #     return False


def update_employee_unavailable_shift(
    db, employee_unavailable_shift_id: int, employee_unavailable_shift_request: EmployeeUnavailableShiftRequest
) -> EmployeeUnavailableShiftResponse:
    employee_unavailable_shift_model = get_employee_unavailable_shift_by_id(db, employee_unavailable_shift_id)
    if employee_unavailable_shift_model:
        try:
            employee_unavailable_shift_model.shift_id = employee_unavailable_shift_request.shift_id
            employee_unavailable_shift_model.employee_id = employee_unavailable_shift_request.employee_id
            employee_unavailable_shift_model.date = employee_unavailable_shift_request.date
            db.commit()
            db.refresh(employee_unavailable_shift_model)
            return employee_unavailable_shift_model
        except SQLAlchemyError as e:
            logger.error(f"Failed to update unavailable shift: {e}")
            db.rollback()
            return None
    return None


def delete_employee_unavailable_shift(db, employee_unavailable_shift_id: int):
    employee_unavailable_shift_model = get_employee_unavailable_shift_by_id(
        db, employee_unavailable_shift_id
    )
    if employee_unavailable_shift_model:
        try:
            db.delete(employee_unavailable_shift_model)
            db.commit()
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete unavailable shift: {e}")
            db.rollback()
            return False
    return False
