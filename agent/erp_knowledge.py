"""
P3.2 — Base de connaissances ERP structurée.

Fournit :
- Définitions des entités ERP standard (Invoice, Product, Customer, etc.)
- Règles métier communes (TVA, audit trail, multi-devise, workflow)
- Patterns d'implémentation recommandés par entité
- Injection automatique dans le contexte du planner quand le goal est ERP-related
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Entités ERP et leurs champs obligatoires
# ---------------------------------------------------------------------------

ERP_ENTITIES: dict[str, dict] = {
    "invoice": {
        "aliases": ["facture", "billing", "facturation", "bill"],
        "required_fields": [
            "id (UUID ou entier auto-incrémenté)",
            "number (str, unique, ex: INV-2024-001)",
            "date (Date)",
            "due_date (Date)",
            "status (enum: draft|sent|paid|cancelled|overdue)",
            "customer_id (FK → Customer)",
            "lines (relation → InvoiceLine[])",
            "subtotal (Decimal)",
            "tax_amount (Decimal)",
            "total (Decimal)",
            "currency (str, ISO 4217, ex: EUR)",
            "notes (str, optionnel)",
            "created_at / updated_at (DateTime)",
        ],
        "business_rules": [
            "Le numéro doit être unique et séquentiel (jamais de gap)",
            "Une facture envoyée ne peut pas être modifiée — créer un avoir (credit note)",
            "Le total = subtotal + tax_amount ; vérifier à chaque sauvegarde",
            "Audit trail : journaliser chaque changement de statut",
            "Support multi-devise : stocker le taux de change au moment de la création",
        ],
        "related_entities": ["Customer", "InvoiceLine", "Product", "Payment"],
        "pattern": "state_machine",
    },
    "product": {
        "aliases": ["article", "item", "produit", "sku"],
        "required_fields": [
            "id",
            "sku (str, unique)",
            "name (str)",
            "description (str)",
            "unit_price (Decimal)",
            "currency (str)",
            "tax_rate (Decimal, ex: 0.20)",
            "stock_quantity (int)",
            "unit_of_measure (str, ex: pcs/kg/liter)",
            "category_id (FK)",
            "is_active (bool)",
            "created_at / updated_at",
        ],
        "business_rules": [
            "Le SKU doit être unique — utiliser un index UNIQUE en DB",
            "Le prix ne peut pas être négatif",
            "Gérer les variations (taille, couleur) via une table ProductVariant",
            "Tracer les mouvements de stock dans StockMovement (entrée/sortie/ajustement)",
        ],
        "related_entities": [
            "Category",
            "ProductVariant",
            "StockMovement",
            "InvoiceLine",
        ],
        "pattern": "catalog",
    },
    "customer": {
        "aliases": ["client", "contact", "partner", "partenaire"],
        "required_fields": [
            "id",
            "name (str)",
            "email (str, unique)",
            "phone (str)",
            "address (str ou relation → Address)",
            "tax_number (str, ex: numéro TVA)",
            "currency (str, devise préférée)",
            "payment_terms (int, jours, ex: 30)",
            "is_active (bool)",
            "created_at / updated_at",
        ],
        "business_rules": [
            "L'email doit être unique et validé",
            "Respecter le RGPD : champ consent_date, droit à l'oubli",
            "Historiser les adresses (ne pas modifier, créer une nouvelle)",
            "Calculer le solde outstanding (somme des factures non payées)",
        ],
        "related_entities": ["Invoice", "Address", "Contact", "Payment"],
        "pattern": "crm_entity",
    },
    "purchase_order": {
        "aliases": ["commande", "bon de commande", "purchase", "po"],
        "required_fields": [
            "id",
            "number (str, unique, ex: PO-2024-001)",
            "date (Date)",
            "expected_delivery (Date)",
            "status (enum: draft|sent|confirmed|received|cancelled)",
            "supplier_id (FK → Supplier)",
            "lines (relation → PurchaseOrderLine[])",
            "total (Decimal)",
            "currency (str)",
            "created_at / updated_at",
        ],
        "business_rules": [
            "Un bon de commande confirmé ne peut pas être annulé sans accord du fournisseur",
            "Réception partielle : mettre à jour les quantités reçues sans clôturer",
            "Générer automatiquement le mouvement de stock à la réception",
            "Three-way match : commande + livraison + facture fournisseur",
        ],
        "related_entities": [
            "Supplier",
            "PurchaseOrderLine",
            "Product",
            "StockMovement",
        ],
        "pattern": "state_machine",
    },
    "employee": {
        "aliases": ["employé", "salarié", "staff", "rh", "hr"],
        "required_fields": [
            "id",
            "first_name (str)",
            "last_name (str)",
            "email (str, unique)",
            "hire_date (Date)",
            "job_title (str)",
            "department_id (FK)",
            "manager_id (FK self-referential)",
            "salary (Decimal)",
            "contract_type (enum: full_time|part_time|contractor)",
            "is_active (bool)",
            "created_at / updated_at",
        ],
        "business_rules": [
            "Ne jamais supprimer un employé — utiliser is_active=False (audit trail)",
            "Le salaire est confidentiel — access control par rôle",
            "Historiser les changements de salaire dans SalaryHistory",
            "Manager auto-référentiel : vérifier les cycles hiérarchiques",
        ],
        "related_entities": ["Department", "SalaryHistory", "Leave", "Payroll"],
        "pattern": "hierarchical_entity",
    },
    "stock": {
        "aliases": ["inventaire", "inventory", "warehouse", "entrepôt", "stock"],
        "required_fields": [
            "id",
            "product_id (FK)",
            "location_id (FK → Warehouse)",
            "quantity (Decimal)",
            "unit_of_measure (str)",
            "last_updated (DateTime)",
            "min_quantity (Decimal)",
        ],
        "business_rules": [
            "Le stock ne peut pas être négatif (sauf si backorder activé)",
            "Chaque mouvement doit être tracé dans StockMovement (FIFO ou FEFO)",
            "Alertes automatiques quand stock < min_quantity",
            "Valorisation du stock : FIFO, LIFO, ou coût moyen pondéré",
        ],
        "related_entities": ["Product", "Warehouse", "StockMovement", "PurchaseOrder"],
        "pattern": "inventory",
    },
}

# ---------------------------------------------------------------------------
# Patterns d'implémentation communs
# ---------------------------------------------------------------------------

ERP_PATTERNS: dict[str, dict] = {
    "state_machine": {
        "description": "Entité avec cycle de vie et transitions d'état",
        "implementation": (
            "Utiliser un champ status avec enum Python.\n"
            "Définir les transitions autorisées dans un dict TRANSITIONS.\n"
            "Lever une ValueError si la transition est invalide.\n"
            "Journaliser chaque transition dans une table *History (audit trail).\n\n"
            "Exemple Python/SQLAlchemy :\n"
            "  TRANSITIONS = {\n"
            "    'draft': ['sent', 'cancelled'],\n"
            "    'sent': ['paid', 'overdue', 'cancelled'],\n"
            "    'paid': [],\n"
            "  }\n"
            "  def transition(self, new_status):\n"
            "    if new_status not in TRANSITIONS[self.status]:\n"
            "        raise ValueError(f'Cannot go from {self.status} to {new_status}')\n"
            "    self.status = new_status"
        ),
    },
    "audit_mixin": {
        "description": "Mixin SQLAlchemy pour l'audit trail automatique",
        "implementation": (
            "class AuditMixin:\n"
            "    created_at = Column(DateTime, default=func.now())\n"
            "    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())\n"
            "    created_by = Column(String(100))\n"
            "    updated_by = Column(String(100))\n\n"
            "Combiner avec SQLAlchemy events pour auto-remplir created_by/updated_by\n"
            "via @event.listens_for(Session, 'before_flush')"
        ),
    },
    "soft_delete": {
        "description": "Suppression logique (is_active / deleted_at) au lieu de DELETE",
        "implementation": (
            "Ne jamais utiliser session.delete() sur des entités ERP.\n"
            "Ajouter un champ deleted_at = Column(DateTime, nullable=True).\n"
            "Filtrer toutes les requêtes avec .where(Model.deleted_at.is_(None)).\n"
            "Utiliser un filter global via query_class ou SQLAlchemy hybrid property."
        ),
    },
    "multi_tenant": {
        "description": "Isolation des données par tenant (SaaS ERP)",
        "implementation": (
            "Ajouter tenant_id (FK → Tenant) sur toutes les tables métier.\n"
            "Utiliser un middleware FastAPI pour extraire le tenant depuis le JWT.\n"
            "Injecter automatiquement tenant_id dans toutes les requêtes via\n"
            "SQLAlchemy event ou un repository base class.\n"
            "Index composite (tenant_id, id) sur toutes les tables."
        ),
    },
    "decimal_money": {
        "description": "Gestion correcte des montants monétaires",
        "implementation": (
            "Toujours utiliser Decimal (Python) ou NUMERIC(15, 4) (SQL).\n"
            "Ne jamais utiliser float pour les montants.\n"
            "Stocker le montant ET la devise (currency code ISO 4217).\n"
            "Stocker le taux de change au moment de la transaction.\n"
            "Arrondir avec ROUND_HALF_UP pour la comptabilité."
        ),
    },
}

# ---------------------------------------------------------------------------
# Règles métier globales ERP
# ---------------------------------------------------------------------------

ERP_BUSINESS_RULES = [
    "Audit trail obligatoire : toutes les modifications doivent être tracées (qui, quand, quoi).",
    "Soft delete partout : ne jamais supprimer physiquement des données comptables ou RH.",
    "Montants en Decimal(15,4), jamais en float — risque d'erreurs d'arrondi.",
    "Numérotation séquentielle sans gap : factures, bons de commande, etc.",
    "Validation côté serveur obligatoire même si validation côté client présente.",
    "Contrôle d'accès RBAC : différencier viewer/editor/admin sur chaque module.",
    "Internationalisation (i18n) : dates ISO 8601, devises ISO 4217, fuseaux horaires UTC.",
    "Transactions DB atomiques pour toutes les opérations multi-tables.",
    "Indexer les colonnes de recherche fréquente : status, date, customer_id, created_at.",
]

# ---------------------------------------------------------------------------
# Détection automatique
# ---------------------------------------------------------------------------


def detect_erp_entities(goal: str) -> list[str]:
    """Retourne les noms d'entités ERP détectées dans le goal."""
    low = goal.lower()
    detected = []
    for entity_name, entity_data in ERP_ENTITIES.items():
        triggers = [entity_name] + entity_data.get("aliases", [])
        if any(t in low for t in triggers):
            detected.append(entity_name)
    return detected


def get_erp_context(goal: str, max_entities: int = 2) -> str:
    """
    Retourne le contexte ERP pertinent pour injecter dans le planner.

    Détecte les entités ERP mentionnées dans le goal et retourne
    leurs champs requis, règles métier, et patterns recommandés.
    """
    entities = detect_erp_entities(goal)[:max_entities]
    if not entities:
        return ""

    sections = [
        "=== Contexte ERP — Règles métier et champs requis ===\n",
        "Règles globales ERP :",
    ]
    for rule in ERP_BUSINESS_RULES[:5]:
        sections.append(f"  • {rule}")

    for entity_name in entities:
        entity = ERP_ENTITIES[entity_name]
        sections.append(f"\nEntité {entity_name.upper()} :")
        sections.append("  Champs requis :")
        for field in entity["required_fields"][:10]:
            sections.append(f"    - {field}")
        sections.append("  Règles métier :")
        for rule in entity["business_rules"][:4]:
            sections.append(f"    • {rule}")

        # Ajouter le pattern recommandé
        pattern_name = entity.get("pattern")
        if pattern_name and pattern_name in ERP_PATTERNS:
            pattern = ERP_PATTERNS[pattern_name]
            sections.append(f"  Pattern recommandé ({pattern_name}) :")
            # Premier paragraphe du pattern seulement
            first_line = pattern["implementation"].split("\n")[0]
            sections.append(f"    {first_line}")

    return "\n".join(sections)
