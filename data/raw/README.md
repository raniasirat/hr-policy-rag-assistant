# data/raw — HR documents to index

Supported: `.pdf`, `.docx`, `.csv`. This README is ignored by the loader.

| File | Content |
|---|---|
| `policy_01_annual_leave.pdf` | Annual leave policy (2 pages) |
| `policy_02_remote_work.pdf` | Remote work policy (2 pages) |
| `policy_03_performance_review.pdf` | Performance review policy (2 pages) |
| `policy_04_code_of_conduct.pdf` | Code of conduct (2 pages) |
| `policy_05_training_development.pdf` | Training and development policy (2 pages) |

The five PDFs are included in this folder and identify AlNoor Technologies as the company. Confirm permission to redistribute them and preserve any required attribution before publishing. No separate license or attribution file was found in the project during the publication audit.

## Employee data (CSV) is restricted

`employees.csv` is **not bundled** in this package. A CSV is treated as sensitive employee data:

- The web API never searches, lists or exposes CSV files unless the request carries the administrator key (`X-Admin-Key`,
  set with `ADMIN_API_KEY` in `.env`). Without that key a CSV is invisible to the chat and to the document list.
- To use it: copy `employees.csv` here (it is git-ignored), set `ADMIN_API_KEY`, run `python scripts/ingest.py`
  (or upload it in the web interface after unlocking administrator access), and unlock it in **Settings**.
- Format: UTF-8, header row, first column = record id (cited as `employees.csv, record EMP001`).
- The command-line chat (`scripts/chat.py`) has **no** access control and would answer from a CSV. Use it with
  employee data only on a computer you alone control.
