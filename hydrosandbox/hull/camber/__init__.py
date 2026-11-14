"""
Hulls from camber (github.com/electricshipwrights/camber), a constant-camber hull
line drawing tool: its documents, its sweep, and its blends of hull families,
ported to Python so a blend (or any authored number) can be a design variable.
"""

from hydrosandbox.hull.camber.document import CamberHull, Station, StationPoint, parse_document
from hydrosandbox.hull.camber.sweep import SweptHull
from hydrosandbox.hull.camber.family import blend, promote_family, mean_value_weights
from hydrosandbox.hull.camber.blended import BlendedHull, HullFamily
