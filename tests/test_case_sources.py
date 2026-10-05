from urllib.parse import urlparse

from plagueshield.data import load_all_cases, load_case
from plagueshield.knowledge.case_sources import case_reference_info


def test_every_case_has_official_references():
    for case in load_all_cases():
        info = case_reference_info(case)
        assert info['source_note']
        assert info['references']
        for reference in info['references']:
            url = urlparse(reference['url'])
            assert url.scheme == 'https'
            assert url.hostname == 'www.who.int' or url.hostname.endswith('.cdc.gov')
            assert reference['title'] and reference['scope']


def test_synthetic_cases_do_not_claim_reported_sources():
    for case in load_all_cases():
        if case.provenance.origin.value == 'synthetic':
            info = case_reference_info(case)
            assert 'not a reported patient' in info['source_note']
            assert all(r['scope'] != 'Outbreak context' for r in info['references'])


def test_unverified_mongolia_citation_is_disclosed():
    info = case_reference_info(load_case('PS-2019-MN-001'))
    assert 'not been verified' in info['source_note']
    assert all(r['scope'] != 'Outbreak context' for r in info['references'])


def test_report_references_use_specific_outbreak_pages():
    for case_id in ['PS-2017-MG-014', 'PS-2020-CD-021']:
        info = case_reference_info(load_case(case_id))
        reports = [r for r in info['references'] if r['scope'] == 'Outbreak context']
        assert len(reports) == 1
        assert '/disease-outbreak-news/item/' in reports[0]['url']
