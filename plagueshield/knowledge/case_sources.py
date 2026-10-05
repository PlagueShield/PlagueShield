"""Official reference context for bundled vignettes, not patient-level proof."""

from copy import deepcopy
from typing import Any

from ..models import CaseRecord

WHO_BACKGROUND = {
    "source": "WHO",
    "title": "Plague: fact sheet",
    "url": "https://www.who.int/news-room/fact-sheets/detail/plague",
    "scope": "Background",
}
CDC_DIAGNOSIS = {
    "source": "CDC",
    "title": "Clinical testing and diagnosis for plague",
    "url": "https://www.cdc.gov/plague/hcp/diagnosis-testing/index.html",
    "scope": "Clinical reference",
}
CDC_DEFINITION = {
    "source": "CDC",
    "title": "Plague (Yersinia pestis): 2020 surveillance case definition",
    "url": "https://ndc.services.cdc.gov/case-definitions/plague-2020/",
    "scope": "Surveillance definition",
}
CDC_SURVEILLANCE = {
    "source": "CDC",
    "title": "Plague: maps and statistics",
    "url": "https://www.cdc.gov/plague/maps-statistics/index.html",
    "scope": "Aggregate surveillance",
}
OUTBREAK_REPORTS = {
    "PS-2017-MG-014": {
        "source": "WHO",
        "title": "Plague - Madagascar (27 November 2017)",
        "url": "https://www.who.int/emergencies/disease-outbreak-news/item/27-november-2017-plague-madagascar-en",
        "scope": "Outbreak context",
    },
    "PS-2020-CD-021": {
        "source": "WHO",
        "title": "Plague - Democratic Republic of the Congo (23 July 2020)",
        "url": "https://www.who.int/emergencies/disease-outbreak-news/item/plague-democratic-republic-of-the-congo",
        "scope": "Outbreak context",
    },
}


def case_reference_info(case: CaseRecord) -> dict[str, Any]:
    references = [CDC_DIAGNOSIS, WHO_BACKGROUND]
    origin = case.provenance.origin.value
    if origin == "synthetic":
        note = "Synthetic test scenario, not a reported patient or outbreak. These links provide official reference guidance only."
        references.append(CDC_DEFINITION)
    elif case.case_id in OUTBREAK_REPORTS:
        note = "Illustrative vignette using reported outbreak context. The WHO report does not establish every clinical or laboratory detail in this case."
        references.insert(0, OUTBREAK_REPORTS[case.case_id])
    elif case.case_id == "PS-2019-MN-001":
        note = "Reconstructed vignette. A specific official report for this 2019 event has not been verified; these links provide general context."
        references.append(CDC_DEFINITION)
    else:
        note = "Representative surveillance vignette, not an individual patient record. These references provide aggregate context and clinical guidance."
        references.insert(0, CDC_SURVEILLANCE)
    return {
        "source_note": note,
        "references": deepcopy(references),
    }
