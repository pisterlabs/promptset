from dataclasses import dataclass, field
from typing import List, Optional
from ojp.guidance_advice_enumeration import GuidanceAdviceEnumeration
from ojp.international_text_structure import InternationalTextStructure
from ojp.path_link_structure import PathLinkStructure
from ojp.situation_full_ref_structure_2 import SituationFullRefStructure2
from ojp.track_section_structure import TrackSectionStructure
from ojp.turn_action_enumeration import TurnActionEnumeration

__NAMESPACE__ = "http://www.vdv.de/ojp"


@dataclass
class PathGuidanceSectionStructure:
    """[an extended definition of a NAVIGATION PATH in TMv6 to include the textual
    navigation instructions] description of a piece of a TRIP.

    May include geographic information, turn instructions and
    accessibility information

    :ivar track_section: LINK PROJECTION on the infrastructure network
        of the TRIP LEG together with time information
    :ivar turn_description: Textual description of a manoeuvre. This
        should imply the information from Manoeuvre, TurnAction, and
        TrackSection.RoadName.
    :ivar guidance_advice: various types of guidance advice given to
        travelle.
    :ivar turn_action: the range of alternative turns that can be
        described.
    :ivar direction_hint: Textual direction hint for better
        understanding, e.g. "follow signs to Hamburg".
    :ivar bearing: Absolute bearing after the described manoeuvre.
    :ivar path_link: Description of the type of accessibility on this
        navigation section.
    :ivar situation_full_ref:
    """
    track_section: Optional[TrackSectionStructure] = field(
        default=None,
        metadata={
            "name": "TrackSection",
            "type": "Element",
            "namespace": "http://www.vdv.de/ojp",
        }
    )
    turn_description: Optional[InternationalTextStructure] = field(
        default=None,
        metadata={
            "name": "TurnDescription",
            "type": "Element",
            "namespace": "http://www.vdv.de/ojp",
        }
    )
    guidance_advice: Optional[GuidanceAdviceEnumeration] = field(
        default=None,
        metadata={
            "name": "GuidanceAdvice",
            "type": "Element",
            "namespace": "http://www.vdv.de/ojp",
        }
    )
    turn_action: Optional[TurnActionEnumeration] = field(
        default=None,
        metadata={
            "name": "TurnAction",
            "type": "Element",
            "namespace": "http://www.vdv.de/ojp",
        }
    )
    direction_hint: Optional[InternationalTextStructure] = field(
        default=None,
        metadata={
            "name": "DirectionHint",
            "type": "Element",
            "namespace": "http://www.vdv.de/ojp",
        }
    )
    bearing: Optional[float] = field(
        default=None,
        metadata={
            "name": "Bearing",
            "type": "Element",
            "namespace": "http://www.vdv.de/ojp",
        }
    )
    path_link: Optional[PathLinkStructure] = field(
        default=None,
        metadata={
            "name": "PathLink",
            "type": "Element",
            "namespace": "http://www.vdv.de/ojp",
        }
    )
    situation_full_ref: List[SituationFullRefStructure2] = field(
        default_factory=list,
        metadata={
            "name": "SituationFullRef",
            "type": "Element",
            "namespace": "http://www.vdv.de/ojp",
        }
    )
