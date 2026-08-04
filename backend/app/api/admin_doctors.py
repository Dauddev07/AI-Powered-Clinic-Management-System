import json
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.core.db import get_db
from app.models.audit_log import AuditLog
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.doctor_leave_date import DoctorLeaveDate
from app.models.doctor_shift import DoctorShift
from app.models.ingestion_log import IngestionLog
from app.models.user import User
from app.schemas.doctor import DoctorListOut, DoctorOut, DoctorStatusUpdate
from app.schemas.doctor_csv import (
    AcceptedRowOut,
    CSVConfirmOut,
    CSVPreviewOut,
    DeactivationSkippedOut,
    HeaderSuggestionOut,
    IngestionLogListOut,
    IngestionLogOut,
    RejectedRowOut,
)
from app.services.doctor_csv import (
    RejectedRow,
    find_confirmed_appointment_conflicts,
    format_conflict_reason,
    validate_csv,
)
from app.services.slots import regenerate_slots_for_clinic, regenerate_slots_for_doctor

router = APIRouter(prefix="/admin/doctors", tags=["admin-doctors"])

MAX_CSV_BYTES = 5 * 1024 * 1024


async def _read_csv_upload(file: UploadFile) -> bytes:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be a .csv")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty")
    if len(content) > MAX_CSV_BYTES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File exceeds the 5MB limit")
    return content


def _parse_header_mapping(raw: str | None) -> dict[str, str]:
    """`header_mapping` is the admin's explicitly-confirmed {original_header:
    canonical_field} decisions from the mapping-confirmation step in the UI. Like
    `file`, the client must resend it on every call (preview and confirm) — nothing
    is ever inferred from a prior request.
    """
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="header_mapping must be valid JSON")
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="header_mapping must be a JSON object mapping original header names to canonical field names",
        )
    return data


def _structural_error_detail(result) -> str:
    parts = []
    if result.missing_columns:
        parts.append(f"missing required column(s): {', '.join(result.missing_columns)}")
    if result.duplicate_mapped_columns:
        parts.append(f"multiple columns mapped onto the same field: {', '.join(result.duplicate_mapped_columns)}")
    return f"CSV is structurally invalid — {'; '.join(parts)}"


def _rejected_detail(rejected: list[RejectedRow]) -> str | None:
    if not rejected:
        return None
    return "\n".join(
        f"row {r.row_number}" + (f" ({r.external_doctor_id})" if r.external_doctor_id else "") + f": {r.reason}"
        for r in rejected
    )


def _write_ingestion_log(
    db: Session,
    clinic_id,
    user_id,
    filename: str,
    rows_accepted: int,
    rows_rejected: int,
    status_: str,
    rejected: list[RejectedRow],
) -> None:
    db.add(
        IngestionLog(
            clinic_id=clinic_id,
            source_type="doctor_csv",
            uploaded_by_user_id=user_id,
            filename=filename,
            rows_accepted=rows_accepted,
            rows_rejected=rows_rejected,
            status=status_,
            detail=_rejected_detail(rejected),
        )
    )


@router.post("/csv/preview", response_model=CSVPreviewOut)
async def preview_csv(
    file: UploadFile = File(...),
    header_mapping: str | None = Form(default=None),
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> CSVPreviewOut:
    """Parses and validates the CSV; writes nothing to the doctor/department tables. Only
    logs to ingestion_logs when the preview found a problem (missing columns, no data rows,
    or rejected rows) — a clean preview with nothing wrong is not logged, so the log isn't
    cluttered with routine previews the admin hasn't even tried to confirm yet.

    Headers that don't exactly match a canonical column name are matched against the
    canonical list via the shared embedding model before anything else runs. If any
    header still needs a mapping decision (a confident suggestion awaiting admin
    confirmation, or no confident match at all), this returns immediately with
    needs_header_mapping=True and no validation is attempted — the caller re-submits
    with `header_mapping` populated once the admin has confirmed each suggestion.
    """
    content = await _read_csv_upload(file)
    confirmed_mapping = _parse_header_mapping(header_mapping)
    clinic_id = current_user.clinic_id
    clinic = db.get(Clinic, clinic_id)

    result = validate_csv(
        content, db, clinic_id, clinic.timezone, clinic.default_slot_minutes, confirmed_mapping
    )

    if result.needs_header_mapping:
        return CSVPreviewOut(
            filename=file.filename,
            needs_header_mapping=True,
            header_suggestions=[
                HeaderSuggestionOut(
                    original_header=s.original_header,
                    suggested_field=s.suggested_field,
                    confidence=s.confidence,
                )
                for s in result.header_suggestions
            ],
            unrecognized_headers=result.unrecognized_headers,
        )

    if not result.is_structurally_valid:
        _write_ingestion_log(db, clinic_id, current_user.id, file.filename, 0, 0, "preview_failed", [])
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_structural_error_detail(result),
        )

    if result.rejected:
        _write_ingestion_log(
            db, clinic_id, current_user.id, file.filename,
            len(result.accepted), len(result.rejected), "preview_failed", result.rejected,
        )
        db.commit()
    elif not result.accepted:
        _write_ingestion_log(db, clinic_id, current_user.id, file.filename, 0, 0, "preview_failed", [])
        db.commit()

    accepted_rows = [
        AcceptedRowOut(
            row_number=r.row_number,
            external_doctor_id=r.external_doctor_id,
            full_name=r.full_name,
            department_name=r.department_name,
            department_will_be_created=not r.department_exists,
            specialization=r.specialization,
            is_active=r.is_active,
            shift_days=sorted({s.weekday for s in r.shifts}),
            leave_dates_count=len(r.leave_dates),
        )
        for r in result.accepted
    ]
    rejected_rows = [
        RejectedRowOut(row_number=r.row_number, external_doctor_id=r.external_doctor_id, reason=r.reason)
        for r in result.rejected
    ]

    return CSVPreviewOut(
        filename=file.filename,
        total_rows=len(result.accepted) + len(result.rejected),
        accepted_count=len(result.accepted),
        rejected_count=len(result.rejected),
        accepted_rows=accepted_rows,
        rejected_rows=rejected_rows,
        warnings=result.warnings,
    )


@router.post("/csv/confirm", response_model=CSVConfirmOut)
async def confirm_csv(
    file: UploadFile = File(...),
    header_mapping: str | None = Form(default=None),
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> CSVConfirmOut:
    """Re-validates the freshly uploaded file from scratch (never trusts a prior preview),
    then upserts doctors/shifts/leave-dates for this clinic in a single transaction. If any
    row fails validation, nothing is written except the failure record in ingestion_logs.

    `header_mapping` must be resent here too — the same admin-confirmed decisions used
    for the final preview. If any header still needs a mapping decision at this point
    (e.g. the client never resent it, or resent an incomplete one), nothing is written;
    the admin must go back through preview and confirm the mapping first.
    """
    content = await _read_csv_upload(file)
    confirmed_mapping = _parse_header_mapping(header_mapping)
    clinic_id = current_user.clinic_id
    clinic = db.get(Clinic, clinic_id)

    result = validate_csv(
        content, db, clinic_id, clinic.timezone, clinic.default_slot_minutes, confirmed_mapping
    )

    if result.needs_header_mapping:
        _write_ingestion_log(db, clinic_id, current_user.id, file.filename, 0, 0, "failed", [])
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV headers still need mapping confirmation. Re-run preview and confirm each suggested mapping first.",
        )

    if not result.is_structurally_valid:
        _write_ingestion_log(db, clinic_id, current_user.id, file.filename, 0, 0, "failed", [])
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_structural_error_detail(result),
        )

    if result.rejected:
        _write_ingestion_log(
            db, clinic_id, current_user.id, file.filename,
            len(result.accepted), len(result.rejected), "failed", result.rejected,
        )
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{len(result.rejected)} row(s) failed validation. Nothing was written. Fix the CSV and re-upload.",
        )

    if not result.accepted:
        _write_ingestion_log(db, clinic_id, current_user.id, file.filename, 0, 0, "failed", [])
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV contains no data rows.",
        )

    try:
        department_cache: dict[str, Department] = {
            d.name.strip().lower(): d
            for d in db.execute(select(Department).where(Department.clinic_id == clinic_id)).scalars().all()
        }

        inserted = 0
        updated = 0
        incoming_ids = {row.external_doctor_id for row in result.accepted}

        for row in result.accepted:
            dept_key = row.department_name.strip().lower()
            dept = department_cache.get(dept_key)
            if dept is None:
                dept = Department(clinic_id=clinic_id, name=row.department_name)
                db.add(dept)
                db.flush()
                department_cache[dept_key] = dept

            doctor = db.execute(
                select(Doctor).where(
                    Doctor.clinic_id == clinic_id, Doctor.external_doctor_id == row.external_doctor_id
                )
            ).scalar_one_or_none()

            if doctor is None:
                doctor = Doctor(
                    clinic_id=clinic_id,
                    department_id=dept.id,
                    external_doctor_id=row.external_doctor_id,
                    full_name=row.full_name,
                    specialization=row.specialization,
                    is_active=row.is_active,
                )
                db.add(doctor)
                db.flush()
                inserted += 1
            else:
                doctor.department_id = dept.id
                doctor.full_name = row.full_name
                doctor.specialization = row.specialization
                doctor.is_active = row.is_active
                updated += 1

            for shift in db.execute(
                select(DoctorShift).where(
                    DoctorShift.clinic_id == clinic_id, DoctorShift.doctor_id == doctor.id
                )
            ).scalars().all():
                db.delete(shift)
            db.flush()
            for shift in row.shifts:
                db.add(
                    DoctorShift(
                        clinic_id=clinic_id,
                        doctor_id=doctor.id,
                        weekday=shift.weekday,
                        start_time_utc=shift.start_time_utc,
                        end_time_utc=shift.end_time_utc,
                        slot_minutes=shift.slot_minutes,
                    )
                )

            # Only ever replace this doctor's *CSV-sourced* leave dates — an admin's
            # manually-added 'admin_block' rows must survive a re-upload untouched.
            for leave in db.execute(
                select(DoctorLeaveDate).where(
                    DoctorLeaveDate.clinic_id == clinic_id,
                    DoctorLeaveDate.doctor_id == doctor.id,
                    DoctorLeaveDate.source == "csv",
                )
            ).scalars().all():
                db.delete(leave)
            db.flush()
            for leave_date in row.leave_dates:
                db.add(
                    DoctorLeaveDate(
                        clinic_id=clinic_id, doctor_id=doctor.id, leave_date_utc=leave_date, source="csv"
                    )
                )

        marked_inactive = 0
        deactivation_skipped: list[DeactivationSkippedOut] = []
        absent_doctors = db.execute(
            select(Doctor).where(
                Doctor.clinic_id == clinic_id,
                Doctor.is_active.is_(True),
                Doctor.external_doctor_id.not_in(incoming_ids),
            )
        ).scalars().all()
        for doc in absent_doctors:
            # Deactivating gives this doctor an empty desired schedule (regeneration's
            # own treatment of an inactive doctor) — so any confirmed appointment they
            # currently hold would be orphaned by it. Leave them active instead of
            # silently proceeding; an admin has to resolve it manually.
            conflicts = find_confirmed_appointment_conflicts(
                db, clinic_id, doc.id, shifts=[], leave_dates=[], clinic_timezone=clinic.timezone
            )
            if conflicts:
                deactivation_skipped.append(
                    DeactivationSkippedOut(
                        external_doctor_id=doc.external_doctor_id,
                        full_name=doc.full_name,
                        reason=format_conflict_reason(conflicts, clinic.timezone),
                    )
                )
                continue
            doc.is_active = False
            marked_inactive += 1

        # Regenerate slots in the SAME transaction — it reads the doctor/shift/leave
        # rows this confirm just wrote (visible via flush, not yet committed), so a
        # regeneration bug rolls back the whole ingestion rather than leaving stale
        # slots against half-applied doctor data.
        regen_stats = regenerate_slots_for_clinic(db, clinic_id, actor_user_id=current_user.id)

        _write_ingestion_log(db, clinic_id, current_user.id, file.filename, len(result.accepted), 0, "success", [])
        db.add(
            AuditLog(
                clinic_id=clinic_id,
                actor_user_id=current_user.id,
                action="doctor_csv_ingest",
                entity_type="doctor",
                entity_id=None,
                metadata_json={
                    "filename": file.filename,
                    "inserted": inserted,
                    "updated": updated,
                    "marked_inactive": marked_inactive,
                    "deactivation_skipped": [d.external_doctor_id for d in deactivation_skipped],
                    "slots_inserted": regen_stats.inserted,
                    "slots_flagged_for_review": regen_stats.flagged_for_review,
                },
            )
        )

        db.commit()
    except Exception:
        db.rollback()
        raise

    return CSVConfirmOut(
        filename=file.filename,
        inserted=inserted,
        updated=updated,
        marked_inactive=marked_inactive,
        rejected_count=0,
        rejected_rows=[],
        slots_inserted=regen_stats.inserted,
        slots_flagged_for_review=regen_stats.flagged_for_review,
        deactivation_skipped=deactivation_skipped,
    )


@router.get("/csv/ingestion-logs", response_model=IngestionLogListOut)
def list_ingestion_logs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> IngestionLogListOut:
    base_filter = (
        IngestionLog.clinic_id == current_user.clinic_id,
        IngestionLog.source_type == "doctor_csv",
    )
    total = db.execute(select(func.count()).select_from(IngestionLog).where(*base_filter)).scalar_one()
    stmt = (
        select(IngestionLog)
        .where(*base_filter)
        .order_by(IngestionLog.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    items = list(db.execute(stmt).scalars().all())
    return IngestionLogListOut(total=total, limit=limit, offset=offset, items=items)


def _serialize_doctor(doctor: Doctor, department: Department) -> DoctorOut:
    return DoctorOut(
        id=doctor.id,
        external_doctor_id=doctor.external_doctor_id,
        full_name=doctor.full_name,
        department_name=department.name,
        specialization=doctor.specialization,
        is_active=doctor.is_active,
    )


@router.get("", response_model=DoctorListOut)
def list_doctors(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> DoctorListOut:
    clinic_id = current_user.clinic_id
    total = db.execute(
        select(func.count()).select_from(Doctor).where(Doctor.clinic_id == clinic_id)
    ).scalar_one()
    stmt = (
        select(Doctor, Department)
        .join(Department, Department.id == Doctor.department_id)
        .where(Doctor.clinic_id == clinic_id)
        .order_by(Department.name.asc(), Doctor.full_name.asc())
        .offset(offset)
        .limit(limit)
    )
    rows = db.execute(stmt).all()
    items = [_serialize_doctor(doctor, department) for doctor, department in rows]
    return DoctorListOut(total=total, limit=limit, offset=offset, items=items)


@router.patch("/{doctor_id}/status", response_model=DoctorOut)
def update_doctor_status(
    doctor_id: uuid.UUID,
    payload: DoctorStatusUpdate,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> DoctorOut:
    """Deactivating gives the doctor an empty desired schedule — the exact same
    treatment regeneration and the CSV-absence path already give an inactive
    doctor — so it's blocked by the identical find_confirmed_appointment_conflicts
    check those paths use, never a separate/weaker rule. Either direction
    immediately regenerates this doctor's slots, so a deactivation stops showing
    bookable slots right away and a reactivation's normal shifts start producing
    them again right away — neither has to wait on a CSV upload or the daily
    scheduled job.
    """
    clinic_id = current_user.clinic_id
    doctor = db.execute(
        select(Doctor).where(Doctor.clinic_id == clinic_id, Doctor.id == doctor_id)
    ).scalar_one_or_none()
    if doctor is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Doctor not found")

    clinic = db.get(Clinic, clinic_id)
    was_active = doctor.is_active

    if not payload.is_active and was_active:
        conflicts = find_confirmed_appointment_conflicts(
            db, clinic_id, doctor.id, shifts=[], leave_dates=[], clinic_timezone=clinic.timezone
        )
        if conflicts:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=format_conflict_reason(conflicts, clinic.timezone),
            )

    doctor.is_active = payload.is_active

    regen_stats = None
    if payload.is_active != was_active:
        db.flush()
        regen_stats = regenerate_slots_for_doctor(
            db, clinic_id, doctor.id, clinic.timezone, actor_user_id=current_user.id
        )

    db.add(
        AuditLog(
            clinic_id=clinic_id,
            actor_user_id=current_user.id,
            action="doctor_status_updated",
            entity_type="doctor",
            entity_id=doctor.id,
            metadata_json={
                "external_doctor_id": doctor.external_doctor_id,
                "is_active": doctor.is_active,
                "slots_inserted": regen_stats.inserted if regen_stats else 0,
                "slots_blocked": regen_stats.blocked if regen_stats else 0,
                "slots_unblocked": regen_stats.unblocked if regen_stats else 0,
                "slots_flagged_for_review": regen_stats.flagged_for_review if regen_stats else 0,
            },
        )
    )

    db.commit()
    db.refresh(doctor)

    department = db.get(Department, doctor.department_id)
    return _serialize_doctor(doctor, department)


