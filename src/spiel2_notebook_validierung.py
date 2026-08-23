from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


NOTEBOOK_RECHENZELLEN = (7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
HISTORIE_KEYS = (
    "lagerbestand_fertigerzeugnisse_historie",
    "lagerbestand_einsatzstoffe_historie",
    "fertigungspersonal_historie",
    "liquiditaet_historie",
    "gewinn_verlustvortrag_historie",
    "grossabnehmer_umsatz_historie",
    "_spiel2_bekanntheit_cache",
)


class _DummyDisplay:
    def __call__(self, *args, **kwargs):
        return None


class _DummyIPython:
    def run_line_magic(self, *args, **kwargs):
        return None


def _notebook_cells(notebook_path: Path):
    with Path(notebook_path).open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)
    return [
        "".join(cell.get("source", [])) if cell.get("cell_type") == "code" else ""
        for cell in notebook["cells"]
    ]


def _source_mit_auswahl(source: str, spiel_ordner: str, periode: int, unternehmen: str) -> str:
    source = re.sub(r'AUSWAHL_PERIODE\s*=\s*\d+', f"AUSWAHL_PERIODE = {int(periode)}", source)
    source = re.sub(r'AUSWAHL_SPIEL_ORDNER\s*=\s*"[^"]+"', f'AUSWAHL_SPIEL_ORDNER = "{spiel_ordner}"', source)
    source = re.sub(r'AUSWAHL_UNTERNEHMEN\s*=\s*"[^"]+"', f'AUSWAHL_UNTERNEHMEN = "{unternehmen}"', source)
    return source


def _exec_notebook_fall(ns: dict, code_cells: list[str], spiel_ordner: str, periode: int, unternehmen: str, quiet: bool = True):
    ns["display"] = _DummyDisplay()
    ns["get_ipython"] = lambda: _DummyIPython()
    stdout = io.StringIO()
    sink = contextlib.redirect_stdout(stdout) if quiet else contextlib.nullcontext()
    with sink:
        for cell_idx in NOTEBOOK_RECHENZELLEN:
            source = code_cells[cell_idx]
            if cell_idx == 7:
                source = _source_mit_auswahl(source, spiel_ordner, periode, unternehmen)
            exec(compile(source, f"Kostenrechnung.ipynb cell {cell_idx}", "exec"), ns)
    return stdout.getvalue()


def _bekanntheit_cache_fuer_periode(ns: dict, code_cells: list[str], spiel_ordner: str, periode: int, quiet: bool = True):
    cache = ns.setdefault("_spiel2_bekanntheit_cache", {})
    cache_key = (spiel_ordner, int(periode))
    if cache_key in cache:
        return cache[cache_key]

    stdout = io.StringIO()
    sink = contextlib.redirect_stdout(stdout) if quiet else contextlib.nullcontext()
    with sink:
        exec(compile(code_cells[26], "Kostenrechnung.ipynb cell 26", "exec"), ns)

    bekanntheit_online_df = ns.get("bekanntheit_online_df", pd.DataFrame())
    werte = {}
    if not bekanntheit_online_df.empty:
        for _, row in bekanntheit_online_df.iterrows():
            unternehmen_label = str(row.get("Unternehmen", ""))
            match = re.fullmatch(r"U(\d+)", unternehmen_label)
            if not match:
                continue
            werte[int(match.group(1))] = {
                "berechnet": row.get("Bekanntheit Erfolgswert", row.get("Bekanntheit berechnet", pd.NA)),
                "slb": row.get("Bekanntheit SLB", pd.NA),
            }
    cache[cache_key] = werte
    return werte


def _find_row_by_label(raw: pd.DataFrame, label: str, helpers: dict):
    normalize_text = helpers["normalize_text"]
    safe_iat = helpers["safe_iat"]
    ziel = normalize_text(label)
    for row_idx in range(len(raw)):
        if normalize_text(safe_iat(raw, row_idx, 0, default="")) == ziel:
            return row_idx
    raise ValueError(f"Zeile {label!r} nicht gefunden")


def _read_value_by_label(raw: pd.DataFrame, label: str, col_idx: int, helpers: dict):
    row_idx = _find_row_by_label(raw, label, helpers)
    return helpers["parse_value"](helpers["safe_iat"](raw, row_idx, col_idx))


def _company_report_path(root: Path, spiel_ordner: str, periode: int, unternehmen_idx: int):
    unternehmen_ordner = f"Unternehmen {unternehmen_idx}"
    return (
        Path(root)
        / spiel_ordner
        / f"Periode {int(periode)}"
        / unternehmen_ordner
        / f"Sammelbericht_Periode {int(periode)}_{unternehmen_ordner}_.xls"
    )


def _read_report_values(base_globals: dict, spiel_ordner: str, periode: int, unternehmen_idx: int):
    helpers = {name: base_globals[name] for name in ("read_sheet", "find_sheet_name", "parse_value", "safe_iat", "normalize_text")}
    root = Path(base_globals["ROOT"])
    company_path = _company_report_path(root, spiel_ordner, periode, unternehmen_idx)
    central_path = root / spiel_ordner / "Periode 0" / "Spielleiter" / "Sammelbericht_Periode 0_Spielleiter_.xls"
    if not central_path.exists():
        raise FileNotFoundError(central_path)

    read_sheet = helpers["read_sheet"]
    find_sheet_name = helpers["find_sheet_name"]
    geschaeftsbericht_sheet = find_sheet_name(
        central_path,
        candidates=[f"{7 + 10 * int(periode)}) SLB Geschaeftsbericht"],
        contains="Gesch",
    )
    if geschaeftsbericht_sheet is None:
        raise ValueError(f"Geschaeftsbericht-Sheet nicht gefunden: {central_path}")

    bericht_raw = read_sheet(central_path, sheet_name=geschaeftsbericht_sheet)
    col_idx = unternehmen_idx

    umsatz = _read_value_by_label(bericht_raw, "Umsatzerlose", col_idx, helpers)
    betriebsergebnis = _read_value_by_label(bericht_raw, "= Betriebsergebnis", col_idx, helpers)
    gewinn_vor_steuern = _read_value_by_label(bericht_raw, "= Gewinn vor Steuern", col_idx, helpers)
    steuern = _read_value_by_label(bericht_raw, "- Steuern auf Einkommen und Ertrag", col_idx, helpers)
    periodenueberschuss = _read_value_by_label(bericht_raw, "= Periodenueberschuss/-fehlbetrag", col_idx, helpers)
    eigenkapital = _read_value_by_label(bericht_raw, "Eigenkapital", col_idx, helpers)
    bilanzsumme = _read_value_by_label(bericht_raw, "Bilanzsumme", col_idx, helpers)
    fremdkapitalquote = (bilanzsumme - eigenkapital) / bilanzsumme * 100 if bilanzsumme else pd.NA

    values = {
        "report_datei": str(central_path),
        "report_geschaeftsbericht_sheet": geschaeftsbericht_sheet,
        "report_liquiditaet_sheet": pd.NA,
        "report_umsatz_gesamt": umsatz,
        "report_betriebsergebnis": betriebsergebnis,
        "report_gewinn_vor_steuern": gewinn_vor_steuern,
        "report_steuern": steuern,
        "report_periodenueberschuss": periodenueberschuss,
        "report_eigenkapital": eigenkapital,
        "report_fremdkapitalquote": fremdkapitalquote,
        "report_einzahlungen_umsatz_aktuell": pd.NA,
        "report_einzahlungen_umsatz_vorperiode": pd.NA,
        "report_ueberziehungskredit": pd.NA,
        "report_steuern_liquiditaet": pd.NA,
        "report_zinsen_ueberziehung": pd.NA,
        "report_kassenendbestand": pd.NA,
    }
    if company_path.exists():
        liquiditaet_sheet = find_sheet_name(
            company_path,
            candidates=[f"{12 + 16 * int(periode)}) Liquiditaetsrechnung"],
            contains="Liquid",
        )
        if liquiditaet_sheet is not None:
            liquid_raw = read_sheet(company_path, sheet_name=liquiditaet_sheet)
            values.update({
                "report_liquiditaet_sheet": liquiditaet_sheet,
                "report_einzahlungen_umsatz_aktuell": _read_value_by_label(liquid_raw, "Einzahlungen aus Umsatz Aktuelle Periode", 1, helpers),
                "report_einzahlungen_umsatz_vorperiode": _read_value_by_label(liquid_raw, "Einzahlungen aus Umsatz Vorperiode", 1, helpers),
                "report_ueberziehungskredit": _read_value_by_label(liquid_raw, "Ueberziehungskredit", 1, helpers),
                "report_steuern_liquiditaet": _read_value_by_label(liquid_raw, "Steuern auf Einkommen und Ertrag", 1, helpers),
                "report_zinsen_ueberziehung": _read_value_by_label(liquid_raw, "Zinszahlung Ueberziehungskredit", 1, helpers),
                "report_kassenendbestand": _read_value_by_label(liquid_raw, "Kassenendbestand", 1, helpers),
            })
    return values


def _erfolgswert_aus_notebook(ns: dict, report_values: dict, erfolgswert_row: dict, labels: Iterable[str]):
    beitrag_summe = 0.0
    gewicht_summe = 0.0
    produktqualitaet = 1 + ns["entscheidungen_df"].loc[
        (ns["entscheidungen_df"]["spiel_ordner"] == ns["spiel_ordner"])
        & (ns["entscheidungen_df"]["unternehmen"] == ns["unternehmen"])
        & (ns["entscheidungen_df"]["periode"] <= int(ns["periode"])),
        "produktqualitaet_level",
    ].fillna(0).astype(float).sum()
    inputs = {
        "Umsatz Gesamt": ns.get("umsatzerloese_teur", pd.NA),
        "Umsatzrendite": ns.get("periodenueberschuss_teur", pd.NA) / ns.get("umsatzerloese_teur", 0) * 100 if ns.get("umsatzerloese_teur", 0) else pd.NA,
        "Periodenueberschuss/-fehlbetrag": ns.get("periodenueberschuss_teur", pd.NA),
        "Eigenkapital": report_values["report_eigenkapital"],
        "Fremdkapitalquote": report_values["report_fremdkapitalquote"],
        "Planungsqualitaet": erfolgswert_row.get("input_planungsqualitat", pd.NA),
        "Bekanntheit": ns.get("bekanntheit_index", pd.NA),
        "Produktqualitaet": produktqualitaet,
    }
    for label in labels:
        key = ns["_key_aus_label"](label)
        punkt, _ = ns["_punkt_aus_formel"](label, inputs[label])
        gewicht = erfolgswert_row.get(f"gewicht_{key}", 0.0)
        if pd.notna(punkt):
            beitrag_summe += float(punkt) * float(gewicht)
            gewicht_summe += float(gewicht)
    return max(0.0, beitrag_summe + max(0.0, 1.0 - gewicht_summe) * 1000.0 + float(erfolgswert_row.get("seminarleiterkorrektur", 0.0)))


def _cell27_helper_source(source: str):
    marker = "# Quellencheck fuer CodeBlock 18"
    return source.split(marker)[0] if marker in source else source


def validate_spiel2_notebook(base_globals: dict, notebook_path="Kostenrechnung.ipynb", spiel_limit=None, max_cases=None, unternehmen_liste=None, quiet=True):
    root = Path(base_globals["ROOT"])
    code_cells = _notebook_cells(Path(notebook_path))
    rows = []
    errors = []
    case_count = 0
    folders = list(base_globals["iter_spiel_ordner"](root=root, include_sets=(2,)))
    if spiel_limit is not None:
        folders = folders[: int(spiel_limit)]
    unternehmen_liste = [1] if unternehmen_liste is None else [int(value) for value in unternehmen_liste]

    for folder, meta in folders:
        spiel_ordner = meta["spiel_ordner"]
        ns = dict(base_globals)
        for key in HISTORIE_KEYS:
            ns[key] = {}
        ns["display"] = _DummyDisplay()
        ns["get_ipython"] = lambda: _DummyIPython()
        try:
            # Fuer die Validierung brauchen wir nur die Erfolgswert-Hilfsfunktionen.
            # Bekanntheit selbst wird unten pro Spiel/Periode/Unternehmen durch Block 18 neu berechnet.
            exec(compile(_cell27_helper_source(code_cells[27]), "Kostenrechnung.ipynb cell 27 helpers", "exec"), ns)
        except Exception as exc:
            errors.append({"spiel_ordner": spiel_ordner, "periode": pd.NA, "unternehmen": pd.NA, "fehler": f"Helper: {type(exc).__name__}: {exc}"})
            continue

        for periode in range(1, 7):
            for unternehmen_idx in unternehmen_liste:
                if max_cases is not None and case_count >= int(max_cases):
                    return pd.DataFrame(rows), pd.DataFrame(errors)
                unternehmen = f"Unternehmen_{unternehmen_idx}"
                try:
                    _exec_notebook_fall(ns, code_cells, spiel_ordner, periode, unternehmen, quiet=quiet)
                    report = _read_report_values(base_globals, spiel_ordner, periode, unternehmen_idx)
                    erfolgswert_datei, erfolgswert_sheet = ns["_aktuelles_erfolgswert_sheet"]()
                    erfolgswert_rows = ns["_lese_erfolgswert_blatt"](erfolgswert_datei, periode, erfolgswert_sheet)
                    erfolgswert_row = next(row for row in erfolgswert_rows if row["unternehmen"] == f"U{unternehmen_idx}")
                    bekanntheit_werte = _bekanntheit_cache_fuer_periode(ns, code_cells, spiel_ordner, periode, quiet=quiet)
                    bekanntheit_notebook = bekanntheit_werte.get(unternehmen_idx, {}).get("berechnet", pd.NA)
                    if pd.notna(bekanntheit_notebook):
                        ns["bekanntheit_index"] = bekanntheit_notebook
                    else:
                        ns.pop("bekanntheit_index", None)
                    erfolgswert_notebook = _erfolgswert_aus_notebook(ns, report, erfolgswert_row, ns["ERFOLGSWERT_LABELS"])
                    bekanntheit_report = erfolgswert_row.get("input_bekanntheit", pd.NA)

                    row = {
                        "spiel_ordner": spiel_ordner,
                        "spiel_set": meta["spiel_set"],
                        "spiel_id": meta["spiel_id"],
                        "spiel_global_id": meta["spiel_global_id"],
                        "periode": periode,
                        "unternehmen": f"U{unternehmen_idx}",
                        "notebook_umsatz_gesamt": ns.get("umsatzerloese_teur", pd.NA),
                        "notebook_betriebsergebnis": ns.get("betriebsergebnis_teur", pd.NA),
                        "notebook_gewinn_vor_steuern": ns.get("gewinn_vor_steuern_teur", pd.NA),
                        "notebook_steuern": ns.get("steuern_einkommen_ertrag_teur", pd.NA),
                        "notebook_periodenueberschuss": ns.get("periodenueberschuss_teur", pd.NA),
                        "notebook_einzahlungen_umsatz_aktuell": ns.get("einzahlungen_umsatz_aktuelle_periode_teur", pd.NA),
                        "notebook_einzahlungen_umsatz_vorperiode": ns.get("einzahlungen_umsatz_vorperiode_teur", pd.NA),
                        "notebook_ueberziehungskredit": ns.get("ueberziehungskredit_teur", pd.NA),
                        "notebook_steuern_liquiditaet": ns.get("steuern_zahlung_teur", pd.NA),
                        "notebook_zinsen_ueberziehung": ns.get("zinsen_ueberziehungskredit_teur", pd.NA),
                        "notebook_kassenendbestand": ns.get("kassenendbestand_teur", pd.NA),
                        "notebook_bekanntheit": bekanntheit_notebook,
                        "notebook_erfolgswert": erfolgswert_notebook,
                        "report_bekanntheit": bekanntheit_report,
                        "report_erfolgswert": erfolgswert_row.get("erfolgswert_slb", pd.NA),
                        "erfolgswert_berechnet": erfolgswert_notebook,
                        "erfolgswert_slb": erfolgswert_row.get("erfolgswert_slb", pd.NA),
                    }
                    row.update(report)
                    row["datei"] = row.get("report_datei", pd.NA)
                    row["sheet"] = row.get("report_geschaeftsbericht_sheet", pd.NA)
                    for key in (
                        "umsatz_gesamt",
                        "betriebsergebnis",
                        "gewinn_vor_steuern",
                        "steuern",
                        "periodenueberschuss",
                        "einzahlungen_umsatz_aktuell",
                        "einzahlungen_umsatz_vorperiode",
                        "ueberziehungskredit",
                        "steuern_liquiditaet",
                        "zinsen_ueberziehung",
                        "kassenendbestand",
                        "bekanntheit",
                        "erfolgswert",
                    ):
                        left = row.get(f"notebook_{key}", pd.NA)
                        right = row.get(f"report_{key}", pd.NA)
                        row[f"diff_{key}"] = float(left) - float(right) if pd.notna(left) and pd.notna(right) else pd.NA
                    row["abweichung_berechnet_minus_slb"] = row["diff_erfolgswert"]
                    row["abs_abweichung"] = abs(row["diff_erfolgswert"]) if pd.notna(row["diff_erfolgswert"]) else pd.NA
                    rows.append(row)
                    case_count += 1
                except Exception as exc:
                    errors.append({"spiel_ordner": spiel_ordner, "periode": periode, "unternehmen": f"U{unternehmen_idx}", "fehler": f"{type(exc).__name__}: {exc}"})

    return pd.DataFrame(rows), pd.DataFrame(errors)


def summarize_validation(validation_df: pd.DataFrame):
    diff_cols = [col for col in validation_df.columns if col.startswith("diff_")]
    rows = []
    for col in diff_cols:
        serie = pd.to_numeric(validation_df[col], errors="coerce").dropna()
        if serie.empty:
            continue
        rows.append({
            "Kennzahl": col.replace("diff_", ""),
            "Anzahl": int(serie.size),
            "Max Abs Diff": float(serie.abs().max()),
            "Mittlere Abs Diff": float(serie.abs().mean()),
            "Median Abs Diff": float(serie.abs().median()),
        })
    return pd.DataFrame(rows).sort_values("Max Abs Diff", ascending=False)
