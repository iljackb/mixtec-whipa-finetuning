"""
Code composed by Jack Bowers & Claude
2026/10/18

Build the final TEI/XML document for the AILLA MYUC-1042 source, from:
  - the finalized review CSV (columns: start, end, orth, final_ipa,
    "Notes (Jack)", eng, spn, notes, flags -- flags is dropped)
  - contributors.txt (name<TAB>role, comma-separated for multiple roles)

TEI Structural decisions:
  - <u @n @xml:id="u-{start}" @start @end>, no @synch (redundant with
    <u>'s own start/end -- no word-level timing exists for this source).
  - Original orthography: <seg xml:lang="mix" xml:id="T-o-{start}"
    notation="orth-ucsb">. Note "orth-ucsb" specifically -- NOT "orth" or
    "ipa" -- to distinguish this source's orthographic convention.
  - IPA: <seg xml:lang="mix" xml:id="T-ipa-{start}" notation="ipa">.
    MUST be notation="ipa" exactly -- every extraction script in this
    project's pipeline identifies the IPA tier by checking for this exact
    value; anything else means zero tokens get extracted from this file
    with no error raised.
  - English/Spanish glosses: <spanGrp type="annotations"> with
    <span xml:lang="en"/"es" type="translation">, matching the existing
    convention used elsewhere in this corpus.
  - All who were labeled as "Transcriber" in the Contributions metadata on the AILLA page are
    tagged in @resp in the notes. For shorthand, I'm referring to these individuals as 
    "UCSB team" here
  - UCSB team's notes and my (Jack's) own notes: separate <note> elements (NOT
    <span type="note">), sharing type="notes", differing in @xml:lang and
    @resp. No @target (confirmed not needed). UCSB's notes default to
    xml:lang="es" (matching this project's typical practice); My (Jack's) own default to
    xml:lang="en".

Usage:
    python3 myuc_to_tei.py <csv_file> <contributors_file> [output_file] [wav_file]

    Example:
    python3 myuc_to_tei.py finalized_review.csv contributors.txt MYUC-1042.xml MYUC-1042.wav
"""

import sys
import csv
from pathlib import Path
from xml.sax.saxutils import escape

TEI_NS = "http://www.tei-c.org/ns/1.0"

# Initials used for @resp attributes -- extend this as more contributors'
# notes need attribution beyond JS GVB EC and Jack.
RESP_INITIALS = {
    "Jeremías Salazar": "JS",
    "Guillem Belmar Viernes": "GVB",
    "Eric Campbell" : "EC",
    "Jack Bowers": "JB"
    
}


def parse_contributors(path: Path):
    """Parse contributors.txt into a list of (name, [roles]) tuples."""
    contributors = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # skip "name\trole" header row
        for row in reader:
            if not row or len(row) < 2:
                continue
            name = row[0].strip()
            roles = [r.strip() for r in row[1].split(",") if r.strip()]
            contributors.append((name, roles))
    return contributors


def build_resp_stmts(contributors):
    """Build one <respStmt> per contributor, with one <resp> per role."""
    blocks = []
    for name, roles in contributors:
        resp_lines = "\n".join(f"               <resp>{escape(r)}</resp>" for r in roles)
        block = f"""            <respStmt>
{resp_lines}
               <name>{escape(name)}</name>
            </respStmt>"""
        blocks.append(block)
    return "\n".join(blocks)


def build_header(contributors, wav_file):
    resp_stmts = build_resp_stmts(contributors)
    return f"""   <teiHeader>
      <fileDesc>
         <titleStmt>
            <title>IPA transcriptions of {escape(wav_file)}</title>
            <title xml:lang="en">How the ancestors used to celebrate</title>
            <title xml:lang="mix">Nixi ntsìsá'a kue nà yatá vikǒ</title>
            <respStmt>
               <resp>IPA Transcription</resp>
               <resp>Encoding</resp>
               <name xml:id="JB">Jack Bowers</name>
            </respStmt>
{resp_stmts}
         </titleStmt>
         <publicationStmt>
           <availability>
             <p>CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike).</p>
           </availability>
         </publicationStmt>
         <sourceDesc>
            <bibl>
               <title xml:lang="en">How the ancestors used to celebrate

</title> 
               <title xml:lang="es">Cómo hacían las fiestas los ancestros</title>
               <title xml:lang="mix">Nixi ntsìsá'a kue nà yatá vikǒ</title>
               <publisher>The Archive of the Indigenous Languages of Latin America (AILLA), <orgName>University of Texas at Austin</orgName></publisher>
               <date>2022</date>
               <idno type="AILLA">27416</idno>
               <ptr target="https://www.ailla.utexas.org/sets/27416/"/>
            </bibl> 
         </sourceDesc>
      </fileDesc>
      <profileDesc>
         <textClass>
            <keywords>
               <term>Genre: Narrative</term>
               <term>Content type: Primary text</term>
               <term>Transcription and translation</term>
            </keywords>
         </textClass>
         <creation>
            <placeName><settlement type="town">Yucunani</settlement>
            <settlement type="municipality">San Juan Mixtepec</settlement>
            <region>Oaxaca</region> <country>Mexico</country></placeName>
            <date>2022-01-23</date>
         </creation>
      </profileDesc>
   </teiHeader>"""


def get_resp_initials(name_hint):
    return RESP_INITIALS.get(name_hint, name_hint)


def build_utterance(row, n):
    # Normalize timestamps to exactly 3 decimal places -- spreadsheet apps
    # (Numbers/Excel) can silently re-export a cell with excessive trailing
    # zeros depending on its display format, even though the underlying
    # value never actually changed. Converting through float and back to a
    # fixed-precision string strips that artifact regardless of how the
    # value looked in the source CSV.
    start = f"{float(row['start']):.3f}"
    end = f"{float(row['end']):.3f}"
    orth = escape(row.get("orth", "").strip())
    ipa = escape(row.get("final_ipa", "").strip())
    eng = escape(row.get("eng", "").strip())
    spn = escape(row.get("spn", "").strip())
    ucsb_note = row.get("notes", "").strip()
    jack_note = row.get("Notes (Jack)", "").strip()

    u_id = f"u-{start}"
    orth_id = f"T-o-{start}"
    ipa_id = f"T-ipa-{start}"

    lines = [f'      <u n="{n}" xml:id="{u_id}" start="{start}" end="{end}">']
    lines.append(f'         <seg xml:lang="mix" xml:id="{orth_id}" notation="orth-ucsb">{orth}</seg>')
    lines.append(f'         <seg xml:lang="mix" xml:id="{ipa_id}" notation="ipa">{ipa}</seg>')

    if eng or spn or ucsb_note or jack_note:
        lines.append('         <spanGrp type="annotations">')
        if eng:
            lines.append(f'            <span xml:lang="en" type="translation">{eng}</span>')
        if spn:
            lines.append(f'            <span xml:lang="es" type="translation">{spn}</span>')
        if ucsb_note:
            lines.append(f'            <span type="note" xml:lang="es" resp="#JS #GVB #EC">{escape(ucsb_note)}</span>')
        if jack_note:
            lines.append(f'            <span type="note" xml:lang="en" resp="#JB">{escape(jack_note)}</span>')
        lines.append('         </spanGrp>')

    lines.append('      </u>')
    return "\n".join(lines)


def main():
    # Explicit, numbered command-line arguments -- sys.argv[0] is always the
    # script name itself; every argument this script actually reads is given
    # a named constant below, so the association between "argument number N"
    # and "what it means" is stated directly in the code, not left implicit
    # in the order argparse happens to register arguments.
    CSV_FILE_ARG = 1
    CONTRIBUTORS_FILE_ARG = 2
    OUTPUT_FILE_ARG = 3       # optional -- defaults to "MYUC-1042.xml" if omitted
    WAV_FILE_ARG = 4          # optional -- defaults to "MYUC-1042.wav" if omitted

    if len(sys.argv) <= CONTRIBUTORS_FILE_ARG:
        print("Usage: python3 myuc_to_tei.py <csv_file> <contributors_file> [output_file] [wav_file]")
        print(f"  arg {CSV_FILE_ARG}: the finalized review CSV")
        print(f"  arg {CONTRIBUTORS_FILE_ARG}: contributors.txt")
        print(f"  arg {OUTPUT_FILE_ARG} (optional): output TEI filename, default MYUC-1038.xml")
        print(f"  arg {WAV_FILE_ARG} (optional): source .wav filename, default MYUC-1038.wav")
        sys.exit(1)

    csv_file = sys.argv[CSV_FILE_ARG]
    contributors_file = sys.argv[CONTRIBUTORS_FILE_ARG]
    output_file = sys.argv[OUTPUT_FILE_ARG] if len(sys.argv) > OUTPUT_FILE_ARG else "MYUC-1038.xml"
    wav_file = sys.argv[WAV_FILE_ARG] if len(sys.argv) > WAV_FILE_ARG else "MYUC-1038.wav"

    contributors = parse_contributors(Path(contributors_file))
    print(f"Parsed {len(contributors)} contributors:")
    for name, roles in contributors:
        print(f"  {name}: {', '.join(roles)}")

    header = build_header(contributors, wav_file)

    utterances = []
    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for n, row in enumerate(reader, start=1):
            utterances.append(build_utterance(row, n))

    body = "\n".join(utterances)

    tei_doc = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="{TEI_NS}">
{header}
   <text>
      <body>
{body}
      </body>
   </text>
</TEI>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(tei_doc)

    print(f"\nWrote {len(utterances)} utterances to {output_file}")


if __name__ == "__main__":
    main()
