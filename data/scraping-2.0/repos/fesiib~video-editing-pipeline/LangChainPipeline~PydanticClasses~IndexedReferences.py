from pydantic import BaseModel, Field, validator

from typing import Dict, List, Sequence

from LangChainPipeline.PydanticClasses.References import References

class SingleReference(BaseModel):
    """A single reference to a temporal, spatial, edit operation, or parameter in a user's video editing command along with offset within the original command"""

    offset: int = Field(
        ...,
        title="Offset of the reference int he original command",
        description="Offset of the reference int he original command",
    )
    reference: str = Field(
        ...,
        title="The reference (one of temporal, spatial, edit operation, or parameters)",
        description="The reference (one of temporal, spatial, edit operation, or parameters)",
    )

    def __init__(self, offset=0, reference=""):
        super().__init__(
            offset=offset,
            reference=reference,
        )
    
    @classmethod
    def from_object(cls, single_reference):
        return cls(
            offset=single_reference["offset"],
            reference=single_reference["reference"],
        )
    
    @validator("reference")
    def reference_must_be_valid_string(cls, v):
        return v

    @validator("offset")
    def offset_must_be_nonnegative(cls, v):
        if v < 0 or type(v) != int:
            print("ERROR: offset must be nonnegative")
            return 0
        return v

    def get_object(self):
        return {
            "offset": self.offset,
            "reference": self.reference,
        }

class IndexedReferences(BaseModel):
    """References to temporal, spatial, and edit operations in a user's video editing command along with offsets within the original command"""

    temporal_references: List[SingleReference] = Field(..., description="Temporal references")
    temporal_labels: List[str] = Field(..., description="Temporal reference labels")
    spatial_references: List[SingleReference] = Field(..., description="Spatial references")
    spatial_labels: List[str] = Field(..., description="Spatial reference labels")
    edit_references: List[SingleReference] = Field(..., description="Edit operation references")
    edit: List[str] = Field(..., description="Identified edit operations (one of text, image, shape, blur, cut, crop, zoom)")
    textParameters: List[SingleReference] = Field(..., description="Text edit parameter references")
    imageParameters: List[SingleReference] = Field(..., description="Image edit parameter references")
    shapeParameters: List[SingleReference] = Field(..., description="Shape edit parameter references")
    blurParameters: List[SingleReference] = Field(..., description="Blur edit parameter references")
    cutParameters: List[SingleReference] = Field(..., description="Cut edit parameter references")
    cropParameters: List[SingleReference] = Field(..., description="Crop edit parameter references")
    zoomParameters: List[SingleReference] = Field(..., description="Zoom edit parameter references")

    def __init__(
        self,
        temporal_references=[],
        temporal_labels=[],
        spatial_references=[],
        spatial_labels=[],
        edit_references=[],
        edit=[],
        textParameters=[],
        imageParameters=[],
        shapeParameters=[],
        blurParameters=[],
        cutParameters=[],
        cropParameters=[],
        zoomParameters=[],
    ):
        super().__init__(
            temporal_references=temporal_references,
            temporal_labels=temporal_labels,
            spatial_references=spatial_references,
            spatial_labels=spatial_labels,
            edit_references=edit_references,
            edit=edit,
            textParameters=textParameters,
            imageParameters=imageParameters,
            shapeParameters=shapeParameters,
            blurParameters=blurParameters,
            cutParameters=cutParameters,
            cropParameters=cropParameters,
            zoomParameters=zoomParameters,
        )
    
    @classmethod
    def from_object(cls, references):
        return cls(
            temporal_references=[SingleReference.from_object(x) for x in references["temporal_references"]],
            temporal_labels=references["temporal_labels"],
            spatial_references=[SingleReference.from_object(x) for x in references["spatial_references"]],
            spatial_labels=references["spatial_labels"],
            edit_references=[SingleReference.from_object(x) for x in references["edit_references"]],
            edit=references["edit"],
            textParameters=[SingleReference.from_object(x) for x in references["textParameters"]],
            imageParameters=[SingleReference.from_object(x) for x in references["imageParameters"]],
            shapeParameters=[SingleReference.from_object(x) for x in references["shapeParameters"]],
            blurParameters=[SingleReference.from_object(x) for x in references["blurParameters"]],
            cutParameters=[SingleReference.from_object(x) for x in references["cutParameters"]],
            cropParameters=[SingleReference.from_object(x) for x in references["cropParameters"]],
            zoomParameters=[SingleReference.from_object(x) for x in references["zoomParameters"]],
        )


    @validator("temporal_references")
    def check_temporal_references(cls, v):
        if isinstance(v, list):
            return v
        print("ERROR: temporal_references must be a valid list")
        return []

    @validator("temporal_labels")
    def check_temporal_labels(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] in ["position", "transcript", "video", "other"]:
                result.append(v[i])
            else:
                print(f"WARNING: Temporal label {v[i]} is not valid")
                result.append("other")
        return result
    
    @validator("spatial_references")
    def check_spatial_references(cls, v):
        if isinstance(v, list):
            return v
        print("ERROR: spatial_references must be a valid list")
        return []
    
    @validator("spatial_labels")
    def check_spatial_labels(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] in ["visual-dependent", "independent", "other"]:
                result.append(v[i])
            else:
                print(f"WARNING: Spatial label {v[i]} is not valid")
                result.append("other")
        return result
    
    @validator("edit_references")
    def check_edit_references(cls, v):
        if isinstance(v, list):
            return v
        print("ERROR: edit_references must be a valid list")
        return []

    @validator("edit")
    def check_edit(cls, v):
        result = []
        for i in range(len(v)):
            if v[i] in ["text", "image", "shape", "blur", "cut", "crop", "zoom"]:
                result.append(v[i])
            else:
                print(f"WARNING: Edit operation {v[i]} is not valid")
                result.append("other")
        return result
        

    @validator("textParameters")
    def check_textParameters(cls, v):
        if isinstance(v, list):
            return v
        print("ERROR: textParameters must be a valid list")
        return []
    
    @validator("imageParameters")
    def check_imageParameters(cls, v):
        if isinstance(v, list):
            return v
        print("ERROR: imageParameters must be a valid list")
        return []
    
    @validator("shapeParameters")
    def check_shapeParameters(cls, v):
        if isinstance(v, list):
            return v
        print("ERROR: shapeParameters must be a valid list")
        return []
    
    @validator("blurParameters")
    def check_blurParameters(cls, v):
        if isinstance(v, list):
            return v
        print("ERROR: blurParameters must be a valid list")
        return []
    
    @validator("cutParameters")
    def check_cutParameters(cls, v):
        if isinstance(v, list):
            return v
        print("ERROR: cutParameters must be a valid list")
        return []
    
    @validator("cropParameters")
    def check_cropParameters(cls, v):
        if isinstance(v, list):
            return v
        print("ERROR: cropParameters must be a valid list")
        return []
    
    @validator("zoomParameters")
    def check_zoomParameters(cls, v):
        if isinstance(v, list):
            return v
        print("ERROR: zoomParameters must be a valid list")
        return []

    def get_parameters(self):
        return {
            "textParameters": self.textParameters,
            "imageParameters": self.imageParameters,
            "shapeParameters": self.shapeParameters,
            "blurParameters": self.blurParameters,
            "cutParameters": self.cutParameters,
            "cropParameters": self.cropParameters,
            "zoomParameters": self.zoomParameters,
        }

    def get_parameters_short(self):

        return {
            "text": [item.get_object() for item in self.textParameters],
            "image": [item.get_object() for item in self.imageParameters],
            "shape": [item.get_object() for item in self.shapeParameters],
            "blur": [item.get_object() for item in self.blurParameters],
            "cut": [item.get_object() for item in self.cutParameters],
            "crop": [item.get_object() for item in self.cropParameters],
            "zoom": [item.get_object() for item in self.zoomParameters],
        }
    
    def get_simple_references(self):
        return References(
            temporal=[x.reference for x in self.temporal_references],
            temporal_labels=self.temporal_labels,
            spatial=[x.reference for x in self.spatial_references],
            spatial_labels=self.spatial_labels,
            edit=self.edit,
            textParameters=[x.reference for x in self.textParameters],
            imageParameters=[x.reference for x in self.imageParameters],
            shapeParameters=[x.reference for x in self.shapeParameters],
            blurParameters=[x.reference for x in self.blurParameters],
            cutParameters=[x.reference for x in self.cutParameters],
            cropParameters=[x.reference for x in self.cropParameters],
            zoomParameters=[x.reference for x in self.zoomParameters],
        )

    @classmethod
    def get_instance(
        cls,
        temporal_references,
        temporal_labels,
        spatial_references,
        spatial_labels,
        edit_references,
        edit,
        textParameters,
        imageParameters,
        shapeParameters,
        blurParameters,
        cutParameters,
        cropParameters,
        zoomParameters,
    ):
        temporal_references_formatted = []
        for i in range(len(temporal_references)):
            temporal_references_formatted.append(SingleReference(
                offset=str(temporal_references[i][0]),
                reference=temporal_references[i][1],
            ))
        spatial_references_formatted = []
        for i in range(len(spatial_references)):
            spatial_references_formatted.append(SingleReference(
                offset=str(spatial_references[i][0]),
                reference=spatial_references[i][1],
            ))
        edit_references_formatted = []
        for i in range(len(edit_references)):
            edit_references_formatted.append(SingleReference(
                offset=str(edit_references[i][0]),
                reference=edit_references[i][1],
            ))
        textParameters_formatted = []
        for i in range(len(textParameters)):
            textParameters_formatted.append(SingleReference(
                offset=str(textParameters[i][0]),
                reference=textParameters[i][1],
            ))
        imageParameters_formatted = []
        for i in range(len(imageParameters)):
            imageParameters_formatted.append(SingleReference(
                offset=str(imageParameters[i][0]),
                reference=imageParameters[i][1],
            ))
        shapeParameters_formatted = []
        for i in range(len(shapeParameters)):
            shapeParameters_formatted.append(SingleReference(
                offset=str(shapeParameters[i][0]),
                reference=shapeParameters[i][1],
            ))
        blurParameters_formatted = []
        for i in range(len(blurParameters)):
            blurParameters_formatted.append(SingleReference(
                offset=str(blurParameters[i][0]),
                reference=blurParameters[i][1],
            ))
        cutParameters_formatted = []
        for i in range(len(cutParameters)):
            cutParameters_formatted.append(SingleReference(
                offset=str(cutParameters[i][0]),
                reference=cutParameters[i][1],
            ))
        cropParameters_formatted = []
        for i in range(len(cropParameters)):
            cropParameters_formatted.append(SingleReference(
                offset=str(cropParameters[i][0]),
                reference=cropParameters[i][1],
            ))
        zoomParameters_formatted = []
        for i in range(len(zoomParameters)):
            zoomParameters_formatted.append(SingleReference(
                offset=str(zoomParameters[i][0]),
                reference=zoomParameters[i][1],
            ))
        return cls(
            temporal_references=temporal_references_formatted,
            temporal_labels=temporal_labels,
            spatial_references=spatial_references_formatted,
            spatial_labels=spatial_labels,
            edit_references=edit_references_formatted,
            edit=edit,
            textParameters=textParameters_formatted,
            imageParameters=imageParameters_formatted,
            shapeParameters=shapeParameters_formatted,
            blurParameters=blurParameters_formatted,
            cutParameters=cutParameters_formatted,
            cropParameters=cropParameters_formatted,
            zoomParameters=zoomParameters_formatted,
        )
    
    @classmethod
    def get_dummy_instance(cls):
        '''
        Whenever t| -> 10
        he person | -> 20
        engages wi| -> 30
        th the scr| -> 40
        een, draw | -> 50
        a sparklin| -> 60
        g mark nea| -> 70
        r his head| -> 80
        '''
        return cls(
            temporal_references=[
                SingleReference(offset=0, reference="whenever the person engages with the screen"),
            ],
            temporal_labels=[
                "video",
            ],
            spatial_references=[
                SingleReference(offset=67, reference="near his head"),
            ],
            spatial_labels=[
                "visual-dependent",
            ],
            edit_references=[
                SingleReference(offset=45, reference="draw"),
            ],
            edit=[
                "shape",
            ],
            textParameters=[],
            imageParameters=[],
            shapeParameters=[
                SingleReference(offset=52, reference="sparkling"),
                SingleReference(offset=62, reference="mark"),
            ],
            blurParameters=[],
            cutParameters=[],
            cropParameters=[],
            zoomParameters=[],
        )
