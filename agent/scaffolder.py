"""
P3.7 -- Code scaffolding / template generation.

Generates boilerplate files for common patterns:
- FastAPI endpoint
- Django model / view
- React component
- Python module with tests
- Test file for existing module
"""

import os
from typing import Dict

from agent.config import PROJECT_ROOT
from agent.tooling import write_new_file

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_TEMPLATES: Dict[str, Dict[str, str]] = {
    "fastapi-endpoint": {
        "description": "FastAPI router with CRUD endpoints",
        "extension": ".py",
        "template": '''\
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/{name}", tags=["{name}"])


class {Name}Base(BaseModel):
    name: str


class {Name}Create({Name}Base):
    pass


class {Name}Response({Name}Base):
    id: int

    class Config:
        from_attributes = True


@router.get("/", response_model=List[{Name}Response])
async def list_{name}s():
    """List all {name}s."""
    return []


@router.get("/{{item_id}}", response_model={Name}Response)
async def get_{name}(item_id: int):
    """Get a single {name} by ID."""
    raise HTTPException(status_code=404, detail="{Name} not found")


@router.post("/", response_model={Name}Response, status_code=201)
async def create_{name}(data: {Name}Create):
    """Create a new {name}."""
    return {Name}Response(id=1, **data.model_dump())


@router.delete("/{{item_id}}", status_code=204)
async def delete_{name}(item_id: int):
    """Delete a {name}."""
    pass
''',
    },
    "django-model": {
        "description": "Django model with admin registration",
        "extension": ".py",
        "template": '''\
from django.db import models


class {Name}(models.Model):
    """Model for {name}."""

    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "{name}"
        verbose_name_plural = "{name}s"

    def __str__(self):
        return self.name
''',
    },
    "django-view": {
        "description": "Django class-based views (List + Detail + Create)",
        "extension": ".py",
        "template": """\
from django.views.generic import ListView, DetailView, CreateView
from django.urls import reverse_lazy


class {Name}ListView(ListView):
    template_name = "{name}/list.html"
    context_object_name = "{name}s"
    paginate_by = 20


class {Name}DetailView(DetailView):
    template_name = "{name}/detail.html"
    context_object_name = "{name}"


class {Name}CreateView(CreateView):
    template_name = "{name}/form.html"
    fields = ["name", "description"]
    success_url = reverse_lazy("{name}-list")
""",
    },
    "react-component": {
        "description": "React functional component with TypeScript",
        "extension": ".tsx",
        "template": """\
import React from "react";

interface {Name}Props {{
  title?: string;
  children?: React.ReactNode;
}}

export const {Name}: React.FC<{Name}Props> = ({{ title, children }}) => {{
  return (
    <div className="{name}">
      {{title && <h2>{{title}}</h2>}}
      {{children}}
    </div>
  );
}};

export default {Name};
""",
    },
    "python-module": {
        "description": "Python module with docstring and basic structure",
        "extension": ".py",
        "template": '''\
"""
{Name} module.

Provides functionality for {name} operations.
"""

from typing import List, Optional


class {Name}:
    """Main class for {name} operations."""

    def __init__(self):
        self._items: List[str] = []

    def add(self, item: str) -> None:
        """Add an item."""
        self._items.append(item)

    def get_all(self) -> List[str]:
        """Return all items."""
        return list(self._items)

    def find(self, query: str) -> Optional[str]:
        """Find first item matching query."""
        for item in self._items:
            if query.lower() in item.lower():
                return item
        return None
''',
    },
    "test": {
        "description": "pytest test file skeleton",
        "extension": ".py",
        "template": '''\
"""Tests for {name}."""

import pytest


class Test{Name}:
    """Test suite for {Name}."""

    def test_creation(self):
        """Test basic creation."""
        sample = {"name": "{name}"}
        assert sample["name"] == "{name}"

    def test_basic_operation(self):
        """Test basic operation."""
        items = ["alpha", "beta", "gamma"]
        filtered = [x for x in items if "a" in x]
        assert filtered == ["alpha", "beta", "gamma"]

    def test_edge_case(self):
        """Test edge cases."""
        assert [x for x in [] if x] == []
''',
    },
    # -------------------------------------------------------------------------
    # ERP Templates
    # -------------------------------------------------------------------------
    "invoice": {
        "description": "Modèle SQLAlchemy + service de facturation ERP (facture, lignes, TVA)",
        "extension": ".py",
        "template": '''\
"""
Module de facturation — {Name}.

Fournit le modèle de facture, les lignes de facturation et le service associé.
"""

from datetime import datetime, UTC
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from sqlalchemy import (
    Column, Integer, String, Numeric, DateTime, ForeignKey,
    Enum as SAEnum, Text, Boolean,
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class InvoiceStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    CANCELLED = "cancelled"
    OVERDUE = "overdue"


class {Name}Line(Base):
    """Ligne de facture."""

    __tablename__ = "{name}_lines"

    id = Column(Integer, primary_key=True)
    {name}_id = Column(Integer, ForeignKey("{name}s.id"), nullable=False)
    description = Column(String(500), nullable=False)
    quantity = Column(Numeric(10, 3), nullable=False, default=1)
    unit_price = Column(Numeric(12, 4), nullable=False)
    vat_rate = Column(Numeric(5, 2), nullable=False, default=Decimal("20.00"))
    discount_rate = Column(Numeric(5, 2), nullable=False, default=Decimal("0.00"))

    @property
    def subtotal_ht(self) -> Decimal:
        return self.quantity * self.unit_price * (1 - self.discount_rate / 100)

    @property
    def vat_amount(self) -> Decimal:
        return self.subtotal_ht * self.vat_rate / 100

    @property
    def total_ttc(self) -> Decimal:
        return self.subtotal_ht + self.vat_amount


class {Name}(Base):
    """Facture principale."""

    __tablename__ = "{name}s"

    id = Column(Integer, primary_key=True)
    number = Column(String(50), unique=True, nullable=False)
    status = Column(SAEnum(InvoiceStatus), default=InvoiceStatus.DRAFT, nullable=False)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)  # multi-tenant
    issue_date = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    due_date = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, default="")
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(UTC))

    lines = relationship("{Name}Line", backref="{name}", cascade="all, delete-orphan")

    @property
    def total_ht(self) -> Decimal:
        return sum(line.subtotal_ht for line in self.lines)

    @property
    def total_vat(self) -> Decimal:
        return sum(line.vat_amount for line in self.lines)

    @property
    def total_ttc(self) -> Decimal:
        return self.total_ht + self.total_vat

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "number": self.number,
            "status": self.status,
            "client_id": self.client_id,
            "total_ht": float(self.total_ht),
            "total_vat": float(self.total_vat),
            "total_ttc": float(self.total_ttc),
            "issue_date": self.issue_date.isoformat() if self.issue_date else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "lines": [
                {
                    "id": ln.id,
                    "description": ln.description,
                    "quantity": float(ln.quantity),
                    "unit_price": float(ln.unit_price),
                    "vat_rate": float(ln.vat_rate),
                    "total_ttc": float(ln.total_ttc),
                }
                for ln in self.lines
            ],
        }
''',
    },
    "purchase-order": {
        "description": "Bon de commande fournisseur ERP avec workflow de validation",
        "extension": ".py",
        "template": '''\
"""
Module bon de commande — {Name}.
"""

from datetime import datetime, UTC
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    Column, Integer, String, Numeric, DateTime, ForeignKey,
    Enum as SAEnum, Text, Boolean,
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class POStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SENT = "sent"
    RECEIVED = "received"
    CANCELLED = "cancelled"


class {Name}Line(Base):
    __tablename__ = "{name}_lines"

    id = Column(Integer, primary_key=True)
    {name}_id = Column(Integer, ForeignKey("{name}s.id"), nullable=False)
    product_ref = Column(String(100), nullable=False)
    description = Column(String(500), nullable=False)
    quantity_ordered = Column(Numeric(10, 3), nullable=False)
    quantity_received = Column(Numeric(10, 3), default=Decimal("0"))
    unit_price = Column(Numeric(12, 4), nullable=False)

    @property
    def is_fully_received(self) -> bool:
        return self.quantity_received >= self.quantity_ordered


class {Name}(Base):
    __tablename__ = "{name}s"

    id = Column(Integer, primary_key=True)
    number = Column(String(50), unique=True, nullable=False)
    status = Column(SAEnum(POStatus), default=POStatus.DRAFT, nullable=False)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    requested_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    expected_date = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    lines = relationship("{Name}Line", backref="{name}", cascade="all, delete-orphan")

    def approve(self, user_id: int) -> None:
        if self.status != POStatus.PENDING_APPROVAL:
            raise ValueError(f"Cannot approve order in status {self.status}")
        self.approved_by = user_id
        self.approved_at = datetime.now(UTC)
        self.status = POStatus.APPROVED

    @property
    def total_amount(self) -> Decimal:
        return sum(ln.quantity_ordered * ln.unit_price for ln in self.lines)
''',
    },
    "stock-item": {
        "description": "Gestion de stock ERP : article, mouvements, valorisation",
        "extension": ".py",
        "template": '''\
"""
Module stock — {Name}.
"""

from datetime import datetime, UTC
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    Column, Integer, String, Numeric, DateTime, ForeignKey,
    Enum as SAEnum, Text, Boolean, Index,
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class MovementType(str, Enum):
    IN = "in"           # Entrée (achat, retour client)
    OUT = "out"         # Sortie (vente, retour fournisseur)
    ADJUSTMENT = "adj"  # Ajustement d'inventaire
    TRANSFER = "transfer"


class {Name}(Base):
    """Article en stock."""

    __tablename__ = "{name}s"
    __table_args__ = (
        Index("ix_{name}s_ref_company", "reference", "company_id", unique=True),
    )

    id = Column(Integer, primary_key=True)
    reference = Column(String(100), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    unit = Column(String(20), default="unit")
    quantity_on_hand = Column(Numeric(12, 3), default=Decimal("0"))
    quantity_reserved = Column(Numeric(12, 3), default=Decimal("0"))
    reorder_point = Column(Numeric(12, 3), default=Decimal("0"))
    unit_cost = Column(Numeric(12, 4), default=Decimal("0"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    movements = relationship("{Name}Movement", backref="item")

    @property
    def quantity_available(self) -> Decimal:
        return self.quantity_on_hand - self.quantity_reserved

    @property
    def needs_reorder(self) -> bool:
        return self.quantity_available <= self.reorder_point


class {Name}Movement(Base):
    """Mouvement de stock."""

    __tablename__ = "{name}_movements"

    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey("{name}s.id"), nullable=False)
    movement_type = Column(SAEnum(MovementType), nullable=False)
    quantity = Column(Numeric(12, 3), nullable=False)
    unit_cost = Column(Numeric(12, 4), nullable=True)
    reference_doc = Column(String(100), nullable=True)  # N° facture, bon de commande, etc.
    notes = Column(Text, default="")
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
''',
    },
    "crm-contact": {
        "description": "Contact CRM ERP avec historique d'interactions",
        "extension": ".py",
        "template": '''\
"""
Module CRM — Contact {Name}.
"""

from datetime import datetime, UTC
from enum import Enum

from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey,
    Enum as SAEnum, Text, Boolean,
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class ContactType(str, Enum):
    PROSPECT = "prospect"
    CLIENT = "client"
    SUPPLIER = "supplier"
    PARTNER = "partner"


class InteractionType(str, Enum):
    CALL = "call"
    EMAIL = "email"
    MEETING = "meeting"
    NOTE = "note"


class {Name}(Base):
    """Contact CRM."""

    __tablename__ = "{name}s"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    contact_type = Column(SAEnum(ContactType), default=ContactType.PROSPECT)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=True)
    phone = Column(String(30), nullable=True)
    company_name = Column(String(255), nullable=True)
    job_title = Column(String(150), nullable=True)
    address = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(UTC))

    interactions = relationship("{Name}Interaction", backref="contact", order_by="desc({Name}Interaction.created_at)")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "company_name": self.company_name,
            "contact_type": self.contact_type,
        }


class {Name}Interaction(Base):
    """Historique des interactions CRM."""

    __tablename__ = "{name}_interactions"

    id = Column(Integer, primary_key=True)
    contact_id = Column(Integer, ForeignKey("{name}s.id"), nullable=False)
    interaction_type = Column(SAEnum(InteractionType), nullable=False)
    subject = Column(String(255), nullable=False)
    notes = Column(Text, default="")
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
''',
    },
    "employee": {
        "description": "Module RH ERP : employé, contrat, paie",
        "extension": ".py",
        "template": '''\
"""
Module RH — Employé {Name}.
"""

from datetime import date, datetime, UTC
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    Column, Integer, String, Numeric, DateTime, Date, ForeignKey,
    Enum as SAEnum, Text, Boolean,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ContractType(str, Enum):
    CDI = "cdi"
    CDD = "cdd"
    APPRENTISSAGE = "apprentissage"
    FREELANCE = "freelance"


class {Name}(Base):
    """Employé."""

    __tablename__ = "{name}s"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    employee_number = Column(String(50), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    phone = Column(String(30), nullable=True)
    birth_date = Column(Date, nullable=True)
    hire_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=True)
    contract_type = Column(SAEnum(ContractType), nullable=False)
    department = Column(String(100), nullable=True)
    job_title = Column(String(150), nullable=True)
    manager_id = Column(Integer, ForeignKey("{name}s.id"), nullable=True)
    gross_salary = Column(Numeric(12, 2), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def is_on_contract(self) -> bool:
        return self.end_date is None or self.end_date >= date.today()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "employee_number": self.employee_number,
            "full_name": self.full_name,
            "email": self.email,
            "job_title": self.job_title,
            "department": self.department,
            "contract_type": self.contract_type,
            "hire_date": self.hire_date.isoformat() if self.hire_date else None,
        }
''',
    },
    "rbac-model": {
        "description": "Modèle RBAC (Role-Based Access Control) pour ERP multi-tenant",
        "extension": ".py",
        "template": '''\
"""
RBAC — Rôles et permissions pour {Name}.

Implémente un contrôle d'accès basé sur les rôles (RBAC) pour ERP multi-tenant.
"""

from datetime import datetime, UTC
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Table, Boolean, Text,
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

# Table d'association Rôle ↔ Permission
role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
    Column("permission_id", Integer, ForeignKey("permissions.id"), primary_key=True),
)

# Table d'association Utilisateur ↔ Rôle (par tenant)
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
    Column("company_id", Integer, ForeignKey("companies.id"), primary_key=True),
)


class Permission(Base):
    """Permission atomique (ex: invoice.create, stock.read)."""

    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True)
    code = Column(String(100), unique=True, nullable=False)   # ex: "invoice.create"
    module = Column(String(50), nullable=False)               # ex: "invoice"
    action = Column(String(50), nullable=False)               # ex: "create"
    description = Column(Text, default="")

    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")


class Role(Base):
    """Rôle regroupant des permissions."""

    __tablename__ = "roles"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, default="")
    is_system = Column(Boolean, default=False)  # Rôles système non modifiables
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)  # None = global
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")

    def has_permission(self, code: str) -> bool:
        return any(p.code == code for p in self.permissions)


def check_permission(user_id: int, company_id: int, permission_code: str, session) -> bool:
    """Vérifie qu'un utilisateur a une permission dans un tenant donné."""
    from sqlalchemy import select, and_

    stmt = (
        select(Permission)
        .join(role_permissions, Permission.id == role_permissions.c.permission_id)
        .join(Role, Role.id == role_permissions.c.role_id)
        .join(user_roles, and_(
            user_roles.c.role_id == Role.id,
            user_roles.c.user_id == user_id,
            user_roles.c.company_id == company_id,
        ))
        .where(Permission.code == permission_code)
    )
    return session.execute(stmt).first() is not None
''',
    },
    "tenant-middleware": {
        "description": "Middleware FastAPI de résolution du tenant (multi-tenant par company_id)",
        "extension": ".py",
        "template": '''\
"""
Middleware multi-tenant pour {Name}.

Résout le tenant courant à partir du JWT ou du header X-Company-ID.
Injecte company_id dans request.state pour toutes les routes.
"""

from typing import Callable, Optional
from fastapi import FastAPI, Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class TenantMiddleware(BaseHTTPMiddleware):
    """Résout et valide le tenant pour chaque requête."""

    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/login", "/token"}

    def __init__(self, app, get_company_id_fn: Optional[Callable] = None):
        super().__init__(app)
        self._get_company_id = get_company_id_fn or self._default_resolver

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        company_id = self._resolve_company_id(request)
        if company_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant non résolu. Fournissez X-Company-ID ou un JWT valide.",
            )
        request.state.company_id = company_id
        return await call_next(request)

    def _resolve_company_id(self, request: Request) -> Optional[int]:
        # 1. Header explicite
        header_val = request.headers.get("X-Company-ID")
        if header_val and header_val.isdigit():
            return int(header_val)

        # 2. JWT Authorization
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            try:
                import jwt as pyjwt
                import os
                token = auth.split(" ", 1)[1]
                payload = pyjwt.decode(
                    token,
                    os.environ.get("JWT_SECRET_KEY", ""),
                    algorithms=["HS256"],
                )
                cid = payload.get("company_id")
                if isinstance(cid, int):
                    return cid
            except Exception:
                pass

        return None

    @staticmethod
    def _default_resolver(request: Request) -> Optional[int]:
        return None


def get_current_company_id(request: Request) -> int:
    """Dépendance FastAPI pour récupérer le company_id courant."""
    company_id = getattr(request.state, "company_id", None)
    if company_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant non résolu",
        )
    return company_id


def add_tenant_middleware(app: FastAPI) -> None:
    """Enregistre le middleware sur l'application FastAPI."""
    app.add_middleware(TenantMiddleware)
''',
    },
    "workflow-state-machine": {
        "description": "Machine à états pour workflows ERP (devis→commande→livraison→facture)",
        "extension": ".py",
        "template": '''\
"""
Machine à états pour workflow {Name}.

Définit les transitions valides et les hooks avant/après transition.
Usage typique : devis → commande → en préparation → expédié → livré → facturé
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Callable, Dict, List, Optional, Set


@dataclass
class Transition:
    from_state: str
    to_state: str
    trigger: str
    guard: Optional[Callable] = None      # Condition booléenne avant transition
    on_enter: Optional[Callable] = None   # Hook exécuté à l'entrée du nouvel état
    on_exit: Optional[Callable] = None    # Hook exécuté à la sortie de l'ancien état


class WorkflowError(Exception):
    pass


class {Name}Workflow:
    """
    Machine à états pour {name}.

    Exemple d'utilisation:
        wf = {Name}Workflow()
        wf.trigger("submit", context=order)   # draft → pending_approval
        wf.trigger("approve", context=order)  # pending_approval → approved
    """

    INITIAL_STATE = "draft"

    TRANSITIONS: List[Transition] = [
        Transition("draft", "pending_approval", "submit"),
        Transition("pending_approval", "approved", "approve"),
        Transition("pending_approval", "draft", "reject"),
        Transition("approved", "in_progress", "start"),
        Transition("in_progress", "completed", "complete"),
        Transition("in_progress", "cancelled", "cancel"),
        Transition("approved", "cancelled", "cancel"),
        Transition("draft", "cancelled", "cancel"),
    ]

    def __init__(self, initial_state: Optional[str] = None):
        self.state = initial_state or self.INITIAL_STATE
        self.history: List[Dict] = []

    def _get_valid_transitions(self) -> List[Transition]:
        return [t for t in self.TRANSITIONS if t.from_state == self.state]

    def can_trigger(self, trigger: str, context=None) -> bool:
        for t in self._get_valid_transitions():
            if t.trigger == trigger:
                if t.guard and not t.guard(context):
                    return False
                return True
        return False

    def trigger(self, trigger: str, context=None) -> str:
        for t in self._get_valid_transitions():
            if t.trigger != trigger:
                continue
            if t.guard and not t.guard(context):
                raise WorkflowError(
                    f"Guard failed for transition {t.from_state} → {t.to_state}"
                )
            old_state = self.state
            if t.on_exit:
                t.on_exit(context)
            self.state = t.to_state
            if t.on_enter:
                t.on_enter(context)
            self.history.append({
                "from": old_state,
                "to": self.state,
                "trigger": trigger,
                "at": datetime.now(UTC).isoformat(),
            })
            return self.state
        raise WorkflowError(
            f"No valid transition for trigger '{trigger}' from state '{self.state}'"
        )

    @property
    def available_triggers(self) -> Set[str]:
        return {t.trigger for t in self._get_valid_transitions()}
''',
    },
    "celery-task": {
        "description": "Tâche Celery asynchrone pour traitements ERP longs (exports, emails, calculs)",
        "extension": ".py",
        "template": '''\
"""
Tâches Celery pour {Name}.

Traitements asynchrones : exports, notifications, calculs longs.
"""

import logging
import csv
from datetime import datetime, UTC
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from celery import Celery, Task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

# Configuration de l'application Celery
# app = Celery(__name__)  # Importer depuis votre module celery principal
# from myapp.celery import app


class BaseTask(Task):
    """Classe de base avec gestion d'erreurs et retry automatique."""

    abstract = True
    max_retries = 3
    default_retry_delay = 60  # secondes

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(
            "{name}_task failed: %s | task_id=%s | args=%s",
            exc, task_id, args,
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning(
            "{name}_task retry: %s | task_id=%s", exc, task_id
        )


# @app.task(bind=True, base=BaseTask, name="{name}.process")
def process_{name}(self, payload: Dict[str, Any], company_id: int) -> Dict[str, Any]:
    """
    Traitement principal pour {name}.

    Args:
        payload: Données à traiter
        company_id: ID du tenant

    Returns:
        Résultat du traitement avec statut et métriques
    """
    start = datetime.now(UTC)
    logger.info("Starting {name} processing | company_id=%s", company_id)

    try:
        # --- Logique métier ici ---
        result = _do_process(payload, company_id)

        elapsed = (datetime.now(UTC) - start).total_seconds()
        logger.info("{name} completed in %.2fs | company_id=%s", elapsed, company_id)
        return {
            "status": "success",
            "result": result,
            "elapsed_seconds": elapsed,
            "company_id": company_id,
        }

    except Exception as exc:
        logger.exception("{name} error: %s", exc)
        raise self.retry(exc=exc)


def _do_process(payload: Dict[str, Any], company_id: int) -> Any:
    """Logique métier par défaut, immédiatement exploitable."""
    safe_payload = payload if isinstance(payload, dict) else {}
    return {
        "accepted": True,
        "company_id": company_id,
        "item_count": len(safe_payload),
        "keys": sorted(str(k) for k in safe_payload.keys()),
    }


# @app.task(name="{name}.export_csv")
def export_{name}_csv(
    filters: Dict[str, Any],
    company_id: int,
    notify_email: Optional[str] = None,
) -> str:
    """Exporte les données {name} en CSV et notifie par email."""
    logger.info("Exporting {name} CSV | company_id=%s", company_id)
    with NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", suffix=".csv", delete=False
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["filter_key", "filter_value", "company_id"])
        for key, value in sorted((filters or {}).items(), key=lambda x: str(x[0])):
            writer.writerow([str(key), str(value), str(company_id)])
        csv_path = fh.name
    if notify_email:
        logger.info("CSV ready at %s; notify_email=%s", csv_path, notify_email)
    return csv_path
''',
    },
    "api-pagination": {
        "description": "FastAPI router avec pagination, filtering et sorting standardisés",
        "extension": ".py",
        "template": '''\
"""
API paginée pour {Name} — FastAPI avec filtering, sorting, pagination.
"""

from typing import Generic, List, Optional, TypeVar
from fastapi import APIRouter, Depends, Query, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

T = TypeVar("T")

router = APIRouter(prefix="/{name}s", tags=["{name}"])


class PaginatedResponse(BaseModel, Generic[T]):
    """Réponse paginée standard."""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PaginationParams(BaseModel):
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(20, ge=1, le=200, description="Items per page")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class {Name}Filters(BaseModel):
    search: Optional[str] = None
    status: Optional[str] = None
    company_id: Optional[int] = None
    created_after: Optional[str] = None
    created_before: Optional[str] = None


class {Name}Response(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


def paginate(query, params: PaginationParams) -> dict:
    """Applique la pagination à une query SQLAlchemy."""
    total = query.count()
    items = query.offset(params.offset).limit(params.page_size).all()
    total_pages = (total + params.page_size - 1) // params.page_size
    return {
        "items": items,
        "total": total,
        "page": params.page,
        "page_size": params.page_size,
        "total_pages": total_pages,
        "has_next": params.page < total_pages,
        "has_prev": params.page > 1,
    }


@router.get("/", response_model=PaginatedResponse[{Name}Response])
async def list_{name}s(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
):
    """
    Liste les {name}s avec pagination, filtering et sorting.

    - **page**: Numéro de page (commence à 1)
    - **page_size**: Nombre d'éléments par page (max 200)
    - **search**: Recherche textuelle
    - **sort_by**: Champ de tri
    - **sort_order**: asc ou desc
    """
    sample_items = [
        {"id": 1, "name": "{Name} Alpha"},
        {"id": 2, "name": "{Name} Beta"},
        {"id": 3, "name": "{Name} Gamma"},
    ]
    if search:
        low = search.lower()
        sample_items = [x for x in sample_items if low in x["name"].lower()]
    reverse = sort_order.lower() == "desc"
    if sort_by == "name":
        sample_items = sorted(sample_items, key=lambda x: x["name"], reverse=reverse)
    else:
        sample_items = sorted(sample_items, key=lambda x: x["id"], reverse=reverse)

    total = len(sample_items)
    start = (page - 1) * page_size
    end = start + page_size
    current = sample_items[start:end]
    total_pages = (total + page_size - 1) // page_size if total else 0

    return PaginatedResponse(
        items=current,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1 and total_pages > 0,
    )


@router.get("/{item_id}", response_model={Name}Response)
async def get_{name}(item_id: int):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="{Name} not found")
''',
    },
    "audit-mixin": {
        "description": "Mixin SQLAlchemy d'audit trail automatique (created_at, updated_at, created_by, deleted_at)",
        "extension": ".py",
        "template": '''\
"""
Mixin d'audit trail pour modèles ERP.

Ajoute automatiquement : created_at, updated_at, created_by, updated_by, deleted_at (soft-delete).

Usage:
    class Invoice(AuditMixin, Base):
        __tablename__ = "invoices"
        id = Column(Integer, primary_key=True)
        ...
"""

from datetime import datetime, UTC
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, ForeignKey, Boolean, event
from sqlalchemy.orm import declared_attr, Session


class AuditMixin:
    """Mixin fournissant les colonnes d'audit standard."""

    @declared_attr
    def created_at(cls):
        return Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)

    @declared_attr
    def updated_at(cls):
        return Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(UTC),
            onupdate=lambda: datetime.now(UTC),
            nullable=False,
        )

    @declared_attr
    def created_by(cls):
        return Column(Integer, ForeignKey("users.id"), nullable=True)

    @declared_attr
    def updated_by(cls):
        return Column(Integer, ForeignKey("users.id"), nullable=True)

    @declared_attr
    def deleted_at(cls):
        return Column(DateTime(timezone=True), nullable=True)

    @declared_attr
    def is_deleted(cls):
        return Column(Boolean, default=False, nullable=False)

    def soft_delete(self, deleted_by: Optional[int] = None) -> None:
        """Marque l'enregistrement comme supprimé (soft-delete)."""
        self.deleted_at = datetime.now(UTC)
        self.is_deleted = True
        if deleted_by is not None:
            self.updated_by = deleted_by

    def restore(self) -> None:
        """Restaure un enregistrement soft-deleted."""
        self.deleted_at = None
        self.is_deleted = False


class SoftDeleteMixin(AuditMixin):
    """Étend AuditMixin avec un filtre automatique des soft-deletes."""

    @classmethod
    def active(cls, session: Session):
        """Retourne une query filtrée sur les enregistrements non supprimés."""
        return session.query(cls).filter(cls.is_deleted.is_(False))
''',
    },
}

# ---------------------------------------------------------------------------
# Custom templates from .stella/templates/
# ---------------------------------------------------------------------------


def _load_custom_templates() -> Dict[str, Dict[str, str]]:
    """Load user-defined templates from .stella/templates/."""
    templates_dir = os.path.join(PROJECT_ROOT, ".stella", "templates")
    if not os.path.isdir(templates_dir):
        return {}
    custom = {}
    for fname in os.listdir(templates_dir):
        if not fname.endswith((".py", ".tsx", ".ts", ".js", ".html")):
            continue
        tpl_name = os.path.splitext(fname)[0]
        ext = os.path.splitext(fname)[1]
        path = os.path.join(templates_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            custom[tpl_name] = {
                "description": f"Custom template: {tpl_name}",
                "extension": ext,
                "template": content,
            }
        except OSError:
            pass
    return custom


def list_templates() -> Dict[str, str]:
    """Return available template types with descriptions."""
    all_tpls = {**_TEMPLATES, **_load_custom_templates()}
    return {name: tpl["description"] for name, tpl in all_tpls.items()}


def scaffold(template_type: str, name: str, output_dir: str = "") -> str:
    """Generate a file from a template.

    Args:
        template_type: One of the registered template types
        name: The entity name (e.g., "user", "product", "invoice")
        output_dir: Optional subdirectory to place the file in

    Returns:
        Result message with created file path
    """
    all_tpls = {**_TEMPLATES, **_load_custom_templates()}

    if template_type not in all_tpls:
        available = ", ".join(sorted(all_tpls.keys()))
        return f"[!] Template inconnu : '{template_type}'. Disponibles : {available}"

    tpl = all_tpls[template_type]
    # Prepare template variables
    clean_name = name.strip().replace("-", "_").replace(" ", "_").lower()
    capitalized = "".join(w.capitalize() for w in clean_name.split("_"))

    content = (
        tpl["template"].replace("{name}", clean_name).replace("{Name}", capitalized)
    )

    # Determine output file path
    ext = tpl["extension"]
    if template_type == "test":
        filename = f"test_{clean_name}{ext}"
        if not output_dir:
            output_dir = "tests"
    elif template_type == "react-component":
        filename = f"{capitalized}{ext}"
        if not output_dir:
            output_dir = "src/components"
    else:
        filename = f"{clean_name}{ext}"

    if output_dir:
        rel_path = os.path.join(output_dir, filename)
    else:
        rel_path = filename

    # Resolve collisions robustly:
    # 1) keep requested filename if free
    # 2) fallback to template-prefixed filename
    # 3) if still taken, append numeric suffixes until a free path is found
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.exists(abs_path):
        base_name, ext_name = os.path.splitext(filename)
        candidates = [f"{template_type.replace('-', '_')}_{filename}"]
        candidates.extend(
            [
                f"{template_type.replace('-', '_')}_{base_name}_{i}{ext_name}"
                for i in range(2, 1000)
            ]
        )
        resolved = None
        for cand in candidates:
            cand_rel = os.path.join(output_dir, cand) if output_dir else cand
            cand_abs = os.path.join(PROJECT_ROOT, cand_rel)
            if not os.path.exists(cand_abs):
                resolved = (cand_rel, cand_abs)
                break
        if resolved is None:
            return f"[!] Impossible de trouver un nom libre pour : {filename}"
        rel_path, abs_path = resolved

    result = write_new_file(rel_path, content)
    if result.startswith("ok:"):
        lines = content.count("\n") + 1
        return (
            f"[scaffold] Cree : {rel_path} ({lines} lignes, template '{template_type}')"
        )
    return f"[scaffold] Erreur : {result}"
