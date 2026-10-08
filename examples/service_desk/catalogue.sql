CREATE TABLE datasets (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    steward TEXT NOT NULL,
    sensitivity TEXT NOT NULL
);
INSERT INTO datasets VALUES
 ('Readmissions Extract', 'Admissions followed by a return to hospital within 30 days', 'Quality Improvement', 'de-identified'),
 ('Oncology Registry', 'Cancer diagnoses, staging and treatment dates', 'Cancer Services', 'identifiable'),
 ('Pharmacy Dispensing', 'Medicines dispensed by ward and date', 'Pharmacy', 'de-identified'),
 ('Emergency Department Attendances', 'Arrivals, triage category and time to be seen', 'Emergency Medicine', 'de-identified'),
 ('Theatre Utilisation', 'Scheduled and actual theatre sessions, cancellations and delays', 'Surgical Services', 'aggregate'),
 ('Outpatient Appointments', 'Booked, attended and missed outpatient appointments', 'Outpatient Services', 'de-identified'),
 ('Staffing Rosters', 'Planned and actual staffing by ward and shift', 'Workforce', 'aggregate'),
 ('Patient Experience Survey', 'Responses to the discharge survey, by ward', 'Patient Experience', 'aggregate');
