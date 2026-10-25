"""Reusable manual-route contracts; private review evidence is supplied externally."""
from .route_card import (AnchorSpec, COCO80_NAMES, DummyGeometry, FrameEvidence,
    GeometryConfig, GoalAnnotation, LandmarkEvidence, LocalConnector, Pose,
    RouteCard, build_route_card)
from .bridge import CardBridgeError, CardOracle, load_frozen_card, make_card_oracle_factory

__all__ = ['AnchorSpec', 'COCO80_NAMES', 'DummyGeometry', 'FrameEvidence',
    'GeometryConfig', 'GoalAnnotation', 'LandmarkEvidence', 'LocalConnector',
    'Pose', 'RouteCard', 'build_route_card', 'CardBridgeError', 'CardOracle',
    'load_frozen_card', 'make_card_oracle_factory']
