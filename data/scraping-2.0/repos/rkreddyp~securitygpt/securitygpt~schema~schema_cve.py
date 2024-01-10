### StockJsons
from enum import Enum, IntEnum
from pydantic import Field
from typing import List, Dict, Optional, Type, Union
from securitygpt.schema.schema_base_openai import OpenAISchema


class CVERemediationJson(OpenAISchema):
    """ Remediation information for the CVE.
    """
    patch_remediation: str = Field(..., description="Remediation using patches that are available of the CVE")
    network_remediation: str = Field(..., description="Network level remediations for the CVE, such as WAF")
    host_remediation: str = Field(..., description="Host or operating system level remediation of the CVE")
    application_remediation: str = Field(..., description="Application level remediation or mitigation of the CVE, such as application configuration changes")
    database_remediation: str = Field(..., description="Database level remediation or mitigation of the CVE, such as application configuration changes")
    operating_system_remediation: str = Field(..., description="Operating system level remediation or mitigation of the CVE, for Ubuntu, Amazon Linux etc")
    
class CVEStructureJson(OpenAISchema):
    """ Structured CVE information.
    """
    
    base_score: float = Field(..., description="Base score of the CVE")
    severity: str = Field(..., description="Severity of the CVE")
    attack_vector: str = Field(..., description="Attack vector of the CVE")
    attack_complexity: str = Field(..., description="Attack complexity of the CVE")
    product_name: str = Field(..., description="Product name of the CVE")
    company_name: str = Field(..., description="Company name of the CVE")
    cwe_name: str = Field(..., description="CWE name of the CVE")
    versions_affected: str = Field(..., description="Versions affected of the CVE")
    versions_not_affected: str = Field(..., description="Versions not affected of the CVE")
    applicable_operating_systems: str = Field(..., description="Applicable operating systems of the CVE")   
    application_configuration_needed: str = Field(..., description="Application configuration needed of the CVE to be exploited")
    versions_fixed: str = Field(..., description="Versions fixed of the CVE")
    remediation: CVERemediationJson = Field(..., description="Remediation of the CVE")  
    summary: str = Field(..., description="Summary of the CVE in markdown format")

