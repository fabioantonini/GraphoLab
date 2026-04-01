# Import all models so SQLAlchemy can resolve relationships regardless of import order.
from backend.models.user import User, Organization, Role  # noqa: F401
from backend.models.project import Project, Document, Analysis, ProjectStatus, AnalysisType  # noqa: F401
from backend.models.audit import AuditLog, AuditAction  # noqa: F401
