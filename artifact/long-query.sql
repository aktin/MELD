
--preprocessing of rawdata
CREATE TEMPORARY TABLE temp_joindata AS

--SELECT BASIC INFORMATION
SELECT DISTINCT obs.encounter_num,
                obs.patient_num,
                EXTRACT(years FROM age(vis_dim.start_date, pat_dim.birth_date)) AS age_in_years,
                pat_dim.sex_cd                                                  as gender,
                vis_dim.start_date                                              AS admission_date,
                vis_dim.end_date                                                AS discharge_date
FROM i2b2crcdata.observation_fact obs
         JOIN i2b2crcdata.patient_dimension pat_dim ON (obs.patient_num = pat_dim.patient_num)
         JOIN i2b2crcdata.visit_dimension vis_dim ON (obs.encounter_num = vis_dim.encounter_num)

-- Column Zeitfilter
WHERE vis_dim.start_date BETWEEN :start AND :end;
-- End Column Zeitfilter

CREATE TEMPORARY TABLE temp_sample AS
SELECT (SELECT COUNT(DISTINCT encounter_num) FROM temp_joindata) AS total_aktin_cases;

-- Column P21
ALTER TABLE temp_joindata ADD COLUMN p21_aufnahmedatum TIMESTAMP;
UPDATE temp_joindata SET p21_aufnahmedatum = start_date
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'P21:ADMR%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_joindata.encounter_num;
-- Add a column to store the count of valid cases
ALTER TABLE temp_sample ADD COLUMN total_p21_cases INTEGER;
-- Update the filtered_cases column with the count of distinct encounter_num from temp_joindata
UPDATE temp_sample SET total_p21_cases = (SELECT COUNT(DISTINCT encounter_num) FROM temp_joindata WHERE (p21_aufnahmedatum IS NOT NULL));
-- End Column P21




--FILTER FOR VALID Cases
-- get patient_ide for patient_num
ALTER TABLE temp_joindata ADD COLUMN patient_ide VARCHAR;
UPDATE temp_joindata my SET patient_ide = pat.patient_ide
FROM i2b2crcdata.patient_mapping pat
WHERE my.patient_num = pat.patient_num;

-- get encounter_ide for encounter_num
ALTER TABLE temp_joindata ADD COLUMN encounter_ide VARCHAR;
UPDATE temp_joindata my SET encounter_ide = enc.encounter_ide
FROM i2b2crcdata.encounter_mapping enc
WHERE my.encounter_num = enc.encounter_num;


-- get billing id for encounter
ALTER TABLE temp_joindata ADD COLUMN billing_ide VARCHAR;
UPDATE temp_joindata my SET billing_ide = obs.tval_char
FROM i2b2crcdata.observation_fact obs
WHERE my.encounter_num = obs.encounter_num
AND concept_cd LIKE 'AKTIN:Fall%';


-- ### drop all AKTIN optout from temp_joindata
-- get AKTIN patient_ide
ALTER TABLE temp_joindata ADD COLUMN pat VARCHAR;
UPDATE temp_joindata my SET pat = opt.pat_psn
FROM i2b2crcdata.optinout_patients opt
WHERE ((opt.study_id = 'AKTIN' AND opt.optinout = 'O') OR (opt.study_id = 'CERT' AND opt.optinout = 'I'))
AND opt.pat_ref = 'PAT'
AND opt.pat_psn = my.patient_ide;


-- get AKTIN encounter_ide
ALTER TABLE temp_joindata ADD COLUMN enc VARCHAR;
UPDATE temp_joindata my SET enc = opt.pat_psn
FROM i2b2crcdata.optinout_patients opt
WHERE ((opt.study_id = 'AKTIN' AND opt.optinout = 'O') OR (opt.study_id = 'CERT' AND opt.optinout = 'I'))
AND opt.pat_ref = 'ENC'
AND opt.pat_psn = my.encounter_ide;


-- get AKTIN billing_id
ALTER TABLE temp_joindata ADD COLUMN bil VARCHAR;
UPDATE temp_joindata my SET bil = opt.pat_psn
FROM i2b2crcdata.optinout_patients opt
WHERE ((opt.study_id = 'AKTIN' AND opt.optinout = 'O') OR (opt.study_id = 'CERT' AND opt.optinout = 'I'))
AND opt.pat_ref = 'BIL'
AND opt.pat_psn = my.billing_ide;

-- delete rows merged with optinout_patients
DELETE FROM temp_joindata
WHERE pat IS NOT NULL
OR enc IS NOT NULL
OR bil IS NOT NULL;

-- delete columns
ALTER TABLE temp_joindata DROP COLUMN pat, DROP COLUMN enc, DROP COLUMN bil, DROP COLUMN billing_ide, DROP COLUMN patient_ide, DROP COLUMN encounter_ide;

-- Add a column to store the count of valid cases
ALTER TABLE temp_sample ADD COLUMN valid_aktin_cases INTEGER;
-- Update the filtered_cases column with the count of distinct encounter_num from temp_joindata
UPDATE temp_sample SET valid_aktin_cases = (SELECT COUNT(DISTINCT encounter_num) FROM temp_joindata);

-- Column P21_Fallzahl
-- Add a column to store the count of valid cases
ALTER TABLE temp_sample ADD COLUMN valid_p21_cases INTEGER;
-- Update the filtered_cases column with the count of distinct encounter_num from temp_joindata
UPDATE temp_sample SET valid_p21_cases = (SELECT COUNT(DISTINCT encounter_num) FROM temp_joindata WHERE (p21_aufnahmedatum IS NOT NULL));
-- End Column P21_Fallzahl


-- Column Alterfilter
DELETE FROM temp_joindata
WHERE age_in_years < 18;
-- Add a column to store the count of valid cases
ALTER TABLE temp_sample ADD COLUMN filtered_aktin_cases_age INTEGER;
-- Update the filtered_cases column with the count of distinct encounter_num from temp_joindata
UPDATE temp_sample SET filtered_aktin_cases_age = (SELECT COUNT(DISTINCT encounter_num) FROM temp_joindata);
-- Add a column to store the count of valid cases
ALTER TABLE temp_sample ADD COLUMN filtered_p21_cases_age INTEGER;
-- Update the filtered_cases column with the count of distinct encounter_num from temp_joindata
UPDATE temp_sample SET filtered_p21_cases_age = (SELECT COUNT(DISTINCT encounter_num) FROM temp_joindata WHERE (p21_aufnahmedatum IS NOT NULL));
-- End Column Alterfilter


-- Column Diagnosefilter
DELETE FROM temp_joindata
WHERE encounter_num NOT IN (
    SELECT encounter_num
    FROM i2b2crcdata.observation_fact
    WHERE concept_cd LIKE 'ICD10GM:%' AND provider_id = '@' and modifier_cd = '@'
    AND (concept_cd LIKE 'ICD10GM:%' OR concept_cd LIKE 'ICD10GM:%' OR concept_cd LIKE 'ICD10GM:%')
);
-- Add a column to store the count of valid cases
ALTER TABLE temp_sample ADD COLUMN filtered_aktin_cases_diagnoses INTEGER;
-- Update the filtered_cases column with the count of distinct encounter_num from temp_joindata
UPDATE temp_sample SET filtered_aktin_cases_diagnoses = (SELECT COUNT(DISTINCT encounter_num) FROM temp_joindata);
-- Add a column to store the count of valid cases
ALTER TABLE temp_sample ADD COLUMN filtered_p21_cases_diagnoses INTEGER;
-- Update the filtered_cases column with the count of distinct encounter_num from temp_joindata
UPDATE temp_sample SET filtered_p21_cases_diagnoses = (SELECT COUNT(DISTINCT encounter_num) FROM temp_joindata WHERE (p21_aufnahmedatum IS NOT NULL));
-- End Column Diagnosefilter


-- Column Cedisfilter
DELETE FROM temp_joindata
WHERE encounter_num NOT IN (
    SELECT encounter_num
    FROM i2b2crcdata.observation_fact
    WHERE (concept_cd LIKE 'CEDIS%' OR concept_cd LIKE '75322-8%') AND modifier_cd = '@' AND (concept_cd LIKE '%' OR concept_cd LIKE '%')
);

-- Add a column to store the count of valid cases
ALTER TABLE temp_sample ADD COLUMN filtered_aktin_cases_cedis INTEGER;
-- Update the filtered_cases column with the count of distinct encounter_num from temp_joindata
UPDATE temp_sample SET filtered_aktin_cases_cedis = (SELECT COUNT(DISTINCT encounter_num) FROM temp_joindata);
-- Add a column to store the count of valid cases
ALTER TABLE temp_sample ADD COLUMN filtered_p21_cases_cedis INTEGER;
-- Update the filtered_cases column with the count of distinct encounter_num from temp_joindata
UPDATE temp_sample SET filtered_p21_cases_cedis = (SELECT COUNT(DISTINCT encounter_num) FROM temp_joindata WHERE (p21_aufnahmedatum IS NOT NULL));
-- End Column Cedisfilter

-- Column CedisorIcdfilter
DELETE FROM temp_joindata
WHERE encounter_num NOT IN (
    SELECT encounter_num
    FROM i2b2crcdata.observation_fact
    WHERE ((concept_cd LIKE 'CEDIS%' OR concept_cd LIKE '75322-8%')
           AND modifier_cd = '@'
           AND concept_cd IN ('CEDIS:003', 'CEDIS:004', 'CEDIS:005', 'CEDIS:006', 'CEDIS:007', 'CEDIS:008', 'CEDIS:054',
                              'CEDIS:107', 'CEDIS:151', 'CEDIS:251', 'CEDIS:257', 'CEDIS:352', 'CEDIS:403', 'CEDIS:404',
                              'CEDIS:504', 'CEDIS:651', 'CEDIS:866', 'CEDIS:999')
           OR concept_cd IN ('ICD10GM:I10.01', 'ICD10GM:I10.91', 'ICD10GM:I11.01', 'ICD10GM:I11.91', 'ICD10GM:R03.0'))
    AND provider_id = '@'
);

-- Add a column to store the count of valid cases
ALTER TABLE temp_sample ADD COLUMN filtered_aktin_cases_cedis_icd INTEGER;
-- Update the filtered_cases column with the count of distinct encounter_num from temp_joindata
UPDATE temp_sample SET filtered_aktin_cases_cedis_icd = (SELECT COUNT(DISTINCT encounter_num) FROM temp_joindata);
-- End Column CedisorIcdfilter


-- ### Table 1 : Encounter data (without diagnoses) ###
-- Fallnummer
-- Aufnahmedatum
-- Zeitpunkt der Verlegung/Entlassung (jjjj-mm-dd hh:mm:ss)
-- Alter in Jahren/Monaten
-- Geschlecht

CREATE TEMPORARY TABLE temp_encounter_data AS
SELECT DISTINCT

-- Column Alter_Jahre
age_in_years,
-- End Column Alter_Jahre
-- Column Geschlecht
gender as geschlecht,
-- End Column Geschlecht
-- Column Aufnahmedatum
admission_date as aufnahme_ts,
-- End Column Aufnahmedatum
-- Column Verlegung/Entlassung
discharge_date as entlassung_ts,
-- End Column Verlegung/Entlassung
-- Column patienten_nummer
patient_num,
-- End Column patienten_nummer
encounter_num
FROM temp_joindata
-- Sort table by encounter_num in ascending order
ORDER BY encounter_num ASC;


-- Column Altersgruppen
ALTER TABLE temp_encounter_data ADD COLUMN altersgruppe VARCHAR;
UPDATE temp_encounter_data SET altersgruppe =
CASE
WHEN age_in_years >=0 AND age_in_years < 6 THEN '0-6'
WHEN age_in_years >=7 AND age_in_years < 11 THEN '7-10'
WHEN age_in_years >=11 AND age_in_years < 14 THEN '11-13'
WHEN age_in_years >=14 AND age_in_years < 18 THEN '14-17'
WHEN age_in_years >=18 AND age_in_years < 25 THEN '18-24'
WHEN age_in_years >=25 AND age_in_years < 30 THEN '25-29'
WHEN age_in_years >=30 AND age_in_years < 35 THEN '30-34'
WHEN age_in_years >=35 AND age_in_years < 40 THEN '35-39'
WHEN age_in_years >=40 AND age_in_years < 45 THEN '40-44'
WHEN age_in_years >=45 AND age_in_years < 50 THEN '45-49'
WHEN age_in_years >=50 AND age_in_years < 55 THEN '50-54'
WHEN age_in_years >=55 AND age_in_years < 60 THEN '55-59'
WHEN age_in_years >=60 AND age_in_years < 65 THEN '60-64'
WHEN age_in_years >=65 AND age_in_years < 70 THEN '65-69'
WHEN age_in_years >=70 AND age_in_years < 75 THEN '70-74'
WHEN age_in_years >=75 AND age_in_years < 80 THEN '75-79'
WHEN age_in_years >=80 AND age_in_years < 85 THEN '80-84'
WHEN age_in_years >=85 AND age_in_years < 90 THEN '85-89'
WHEN age_in_years >=90 AND age_in_years < 95 THEN '90-94'
WHEN age_in_years >=95 AND age_in_years < 100 THEN '95-99'
WHEN age_in_years >=100 THEN '100+'
ELSE age_in_years::text
END;
ALTER TABLE temp_encounter_data DROP COLUMN age_in_years;
-- End Column Altersgruppen



-- Column Versicherungsträger
ALTER TABLE temp_encounter_data ADD COLUMN kkname VARCHAR;
UPDATE temp_encounter_data SET kkname = i2b2crcdata.observation_fact.tval_char
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd = 'AKTIN:KKNAME' AND provider_id = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Versicherungsträger

-- Column Versicherungsnummer
ALTER TABLE temp_encounter_data ADD COLUMN iknr VARCHAR;
UPDATE temp_encounter_data SET iknr = i2b2crcdata.observation_fact.tval_char
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd = 'AKTIN:IKNR' AND provider_id = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Versicherungsnummer

-- Column PLZ
ALTER TABLE temp_encounter_data ADD COLUMN plz VARCHAR;
UPDATE temp_encounter_data SET plz = i2b2crcdata.observation_fact.tval_char
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd = 'AKTIN:ZIPCODE' AND provider_id = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column PLZ

-- Column PLZ_kurz
ALTER TABLE temp_encounter_data ADD COLUMN plz_kurz VARCHAR;
UPDATE temp_encounter_data SET plz_kurz = SUBSTR(i2b2crcdata.observation_fact.tval_char,1,3)
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd = 'AKTIN:ZIPCODE' AND provider_id = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column PLZ_kurz

-- Column Zuweisung
ALTER TABLE temp_encounter_data ADD COLUMN zuweisung VARCHAR;
UPDATE temp_encounter_data SET zuweisung = CASE
    WHEN substr(concept_cd, 16) = 'VAP' THEN 'Vertragsarzt/Praxis'
    WHEN substr(concept_cd, 16) = 'KVNPIK' THEN 'KV-Notfallpraxis am Krankenhaus'
    WHEN substr(concept_cd, 16) = 'KVNDAK' THEN 'KV-Notdienst ausserhalb des Krankenhauses'
    WHEN substr(concept_cd, 16) = 'RD' THEN 'Rettungsdienst'
    WHEN substr(concept_cd, 16) = 'NA' THEN 'Notarzt'
    WHEN substr(concept_cd, 16) = 'KLINV' THEN 'Klinik/Verlegung'
    WHEN substr(concept_cd, 16) = 'NPHYS' THEN 'Zuweisung nicht durch Arzt'
    WHEN substr(concept_cd, 16) = 'OTH' THEN 'Andere'
    ELSE
        substr(concept_cd, 16)
    END
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'AKTIN:REFERRAL%' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Zuweisung

-- Column Transport
ALTER TABLE temp_encounter_data ADD COLUMN transport VARCHAR;
UPDATE temp_encounter_data SET transport = CASE
    WHEN concept_cd = 'AKTIN:TRANSPORT:1' THEN 'KTW'
    WHEN concept_cd = 'AKTIN:TRANSPORT:2' THEN 'RTW'
    WHEN concept_cd = 'AKTIN:TRANSPORT:3' THEN 'NAW/NEF/ITW'
    WHEN concept_cd = 'AKTIN:TRANSPORT:4' THEN 'RTH/ITH'
    WHEN concept_cd = 'AKTIN:TRANSPORT:NA' THEN 'Ohne'
    WHEN concept_cd = 'AKTIN:TRANSPORT:OTH' THEN 'Andere'
    ELSE
        concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'AKTIN:TRANSPORT%' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Transport

-- Column Vorstellungsgrund_cedis
ALTER TABLE temp_encounter_data ADD COLUMN cedis VARCHAR;
UPDATE temp_encounter_data SET cedis = CASE
    WHEN concept_cd = '75322-8:UNK' THEN '999'
    WHEN concept_cd = 'CEDIS30:UNK' THEN '999'
    WHEN concept_cd LIKE '75322-8%' THEN substr(concept_cd, 8)
    WHEN concept_cd LIKE 'CEDIS%' THEN substr(concept_cd, 9)
    ELSE
        concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'CEDIS%' OR concept_cd LIKE '75322-8%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Vorstellungsgrund_cedis

-- Column Beschwerden_CEDIS_text
ALTER TABLE temp_encounter_data ADD COLUMN cedis_text VARCHAR;
UPDATE temp_encounter_data SET cedis_text = i2b2crcdata.observation_fact.tval_char
FROM i2b2crcdata.observation_fact
WHERE concept_cd = 'AKTIN:COMPLAINT' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Beschwerden_CEDIS_text

-- Column Symptom_Dauer
ALTER TABLE temp_encounter_data ADD COLUMN symptom_dauer DECIMAL;
UPDATE temp_encounter_data SET symptom_dauer = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd = 'AKTIN:SYMPTOMDURATION' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Symptom_Dauer

-- Column Ersteinschätzung_triage
ALTER TABLE temp_encounter_data ADD COLUMN triage VARCHAR;
UPDATE temp_encounter_data SET triage = CASE
    WHEN concept_cd LIKE 'MTS:%' THEN substr(concept_cd, 5)
    WHEN concept_cd LIKE 'ESI:%' THEN substr(concept_cd, 5)
    WHEN concept_cd LIKE 'AKTIN:ASSESSMENT%' THEN substr(concept_cd, 18)
    ELSE
        concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'MTS%' OR concept_cd LIKE 'ESI%' OR concept_cd LIKE 'AKTIN:ASSESSMENT%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Ersteinschätzung_triage

-- Column Triagesystem
ALTER TABLE temp_encounter_data ADD COLUMN triagesystem VARCHAR;
UPDATE temp_encounter_data SET triagesystem = CASE
    WHEN concept_cd LIKE 'MTS:%' THEN 'MTS'
    WHEN concept_cd LIKE 'ESI:%' THEN 'ESI'
    WHEN concept_cd LIKE 'AKTIN:ASSESSMENT' THEN 'Anderes'
    ELSE
        concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'MTS%' OR concept_cd LIKE 'ESI%' OR concept_cd LIKE 'AKTIN:ASSESSMENT%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Triagesystem

-- Column Atemfrequenz
ALTER TABLE temp_encounter_data ADD COLUMN atemfrequenz DECIMAL;
UPDATE temp_encounter_data SET atemfrequenz = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd = 'LOINC:9279-1' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Atemfrequenz

-- Column O2Saettigung
ALTER TABLE temp_encounter_data ADD COLUMN saettigung DECIMAL;
UPDATE temp_encounter_data SET saettigung = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd = 'LOINC:20564-1' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column O2Saettigung

-- Column Blutdruck_sys
ALTER TABLE temp_encounter_data ADD COLUMN blutdruck_sys DECIMAL;
UPDATE temp_encounter_data SET blutdruck_sys = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd = 'LOINC:8480-6' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Blutdruck_sys

-- Column Herzfrequenz
ALTER TABLE temp_encounter_data ADD COLUMN herzfrequenz DECIMAL;
UPDATE temp_encounter_data SET herzfrequenz = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd = 'LOINC:8867-4' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Herzfrequenz

-- Column Kerntemperatur
ALTER TABLE temp_encounter_data ADD COLUMN kerntemperatur DECIMAL;
UPDATE temp_encounter_data SET kerntemperatur = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd = 'LOINC:8329-5' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Kerntemperatur

-- Column Schmerzskala
ALTER TABLE temp_encounter_data ADD COLUMN schmerzskala DECIMAL;
UPDATE temp_encounter_data SET schmerzskala = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd = 'LOINC:72514-3' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Schmerzskala

-- Column gcs_summe
ALTER TABLE temp_encounter_data ADD COLUMN gcs_summe DECIMAL;
UPDATE temp_encounter_data SET gcs_summe = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd = 'LOINC:9269-2' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column gcs_summe

-- Column gcs_augen
ALTER TABLE temp_encounter_data ADD COLUMN gcs_augen DECIMAL;
UPDATE temp_encounter_data SET gcs_augen = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd = 'LOINC:9267-6' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column gcs_augen

-- Column gcs_verbal
ALTER TABLE temp_encounter_data ADD COLUMN gcs_verbal DECIMAL;
UPDATE temp_encounter_data SET gcs_verbal = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd = 'LOINC:9270-0' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column gcs_verbal

-- Column gcs_motorisch
ALTER TABLE temp_encounter_data ADD COLUMN gcs_motorisch DECIMAL;
UPDATE temp_encounter_data SET gcs_motorisch = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd = 'LOINC:9268-4' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column gcs_motorisch

-- Column pupillenweite_rechts
ALTER TABLE temp_encounter_data ADD COLUMN pupillenweite_rechts  VARCHAR;
UPDATE temp_encounter_data SET pupillenweite_rechts  = CASE
    WHEN substr(concept_cd,12) = 'D' THEN 'weit'
    WHEN substr(concept_cd,12) = 'M' THEN 'mittel'
    WHEN substr(concept_cd,12) = 'C' THEN 'eng'
    WHEN substr(concept_cd,12) = 'UNK' THEN 'unbekannt'
    ELSE
        substr(concept_cd,12)
    END
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'AKTIN:SPPL:%' AND modifier_cd = 'AKTIN:TSITE:R'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column pupillenweite_rechts

-- Column pupillenweite_links
ALTER TABLE temp_encounter_data ADD COLUMN pupillenweite_links  VARCHAR;
UPDATE temp_encounter_data SET pupillenweite_links  = CASE
    WHEN substr(concept_cd,12) = 'D' THEN 'weit'
    WHEN substr(concept_cd,12) = 'M' THEN 'mittel'
    WHEN substr(concept_cd,12) = 'C' THEN 'eng'
    WHEN substr(concept_cd,12) = 'UNK' THEN 'unbekannt'
    ELSE
        substr(concept_cd,12)
    END
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'AKTIN:SPPL:%' AND modifier_cd = 'AKTIN:TSITE:L'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column pupillenweite_links

-- Column pupillenreaktion_rechts
ALTER TABLE temp_encounter_data ADD COLUMN pupillenreaktion_rechts  VARCHAR;
UPDATE temp_encounter_data SET pupillenreaktion_rechts  = CASE
    WHEN substr(concept_cd,12) = 'B' THEN 'prompt'
    WHEN substr(concept_cd,12) = 'D' THEN 'traege'
    WHEN substr(concept_cd,12) = 'A' THEN 'keine'
    WHEN substr(concept_cd,12) = 'UNK' THEN 'unbekannt'
    ELSE
        substr(concept_cd,12)
    END
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'AKTIN:RPPL:%' AND modifier_cd = 'AKTIN:TSITE:R'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column pupillenreaktion_rechts

-- Column pupillenreaktion_links
ALTER TABLE temp_encounter_data ADD COLUMN pupillenreaktion_links  VARCHAR;
UPDATE temp_encounter_data SET pupillenreaktion_links  = CASE
    WHEN substr(concept_cd,12) = 'B' THEN 'prompt'
    WHEN substr(concept_cd,12) = 'D' THEN 'traege'
    WHEN substr(concept_cd,12) = 'A' THEN 'keine'
    WHEN substr(concept_cd,12) = 'UNK' THEN 'unbekannt'
    ELSE
        substr(concept_cd,12)
    END
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'AKTIN:RPPL:%' AND modifier_cd = 'AKTIN:TSITE:L'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column pupillenreaktion_links

-- Column rankin
ALTER TABLE temp_encounter_data ADD COLUMN rankin DECIMAL;
UPDATE temp_encounter_data SET rankin = ROUND(nval_num)
FROM i2b2crcdata.observation_fact
WHERE (concept_cd = 'LOINC:75859-9' OR concept_cd LIKE 'RANKIN:%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column rankin

-- Column Isolation Status
ALTER TABLE temp_encounter_data ADD COLUMN isolation_status VARCHAR;
UPDATE temp_encounter_data
SET isolation_status = CASE
    WHEN of.concept_cd = 'AKTIN:ISOLATION:ISO' THEN 'Isolation'
    WHEN of.concept_cd = 'AKTIN:ISOLATION:RISO' THEN 'Umkehrisolation'
    WHEN of.concept_cd = 'AKTIN:ISOLATION:ISO:NEG' THEN 'keine Isolation'
    ELSE of.concept_cd
END
FROM i2b2crcdata.observation_fact of
WHERE of.concept_cd LIKE 'AKTIN:ISOLATION:%'
AND of.modifier_cd = '@'
AND of.encounter_num = temp_encounter_data.encounter_num;
-- End Column Isolation Status

-- Column Isolation Reason
ALTER TABLE temp_encounter_data ADD COLUMN isolation_grund VARCHAR;
UPDATE temp_encounter_data
SET isolation_grund = CASE
    WHEN of.concept_cd = 'AKTIN:ISOREASON:U80' THEN 'multiresistenter Keim'
    WHEN of.concept_cd = 'AKTIN:ISOREASON:A09.9' THEN 'Gastroenteritis'
    WHEN of.concept_cd = 'AKTIN:ISOREASON:A16.9' THEN 'Tuberkulose'
    WHEN of.concept_cd = 'AKTIN:ISOREASON:G03.9' THEN 'Meningitis'
    WHEN of.concept_cd = 'AKTIN:ISOREASON:OTH' THEN 'Andere'
    ELSE of.concept_cd
END
FROM i2b2crcdata.observation_fact of
WHERE of.concept_cd LIKE 'AKTIN:ISOREASON:%'
AND of.modifier_cd = '@'
AND of.encounter_num = temp_encounter_data.encounter_num;
-- End Column Isolation Reason


-- Column Keime
ALTER TABLE temp_encounter_data ADD COLUMN keime VARCHAR;
UPDATE temp_encounter_data
SET keime = CASE
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:AMRO:NEG' THEN 'keine MRE'
    ELSE of.concept_cd
END
FROM i2b2crcdata.observation_fact of
WHERE of.concept_cd LIKE 'AKTIN:PATHOGENE:AMRO%'
AND of.modifier_cd = '@'
AND of.encounter_num = temp_encounter_data.encounter_num;
-- End Column Keime

-- Column Keime MRSA
ALTER TABLE temp_encounter_data ADD COLUMN keime_mrsa VARCHAR;
UPDATE temp_encounter_data
SET keime_mrsa = CASE
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:MRSA:CONF' THEN 'MRSA'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:MRSA:PB' THEN 'MRSA'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:MRSA:SUSP' THEN 'V.a. MRSA'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:MRSA' THEN 'MRSA'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:MRSA:NEG' THEN 'Kein MRSA'
    ELSE of.concept_cd
END
FROM i2b2crcdata.observation_fact of
WHERE of.concept_cd LIKE 'AKTIN:PATHOGENE:MRSA%'
AND of.modifier_cd = '@'
AND of.encounter_num = temp_encounter_data.encounter_num;
-- End Column Keime MRSA

-- Column Keime 3MRGN
ALTER TABLE temp_encounter_data ADD COLUMN keime_3mrgn VARCHAR;
UPDATE temp_encounter_data
SET keime_3mrgn = CASE
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:3MRGN' THEN '3MRGN'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:3MRGN:CONF' THEN '3MRGN'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:3MRGN:PB' THEN '3MRGN'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:3MRGN:SUSP' THEN 'V.a. 3MRGN'
    ELSE of.concept_cd
END
FROM i2b2crcdata.observation_fact of
WHERE of.concept_cd LIKE 'AKTIN:PATHOGENE:3MRGN%'
AND of.modifier_cd = '@'
AND of.encounter_num = temp_encounter_data.encounter_num;
-- End Column Keime 3MRGN

-- Column Keime 4MRGN
ALTER TABLE temp_encounter_data ADD COLUMN keime_4mrgn VARCHAR;
UPDATE temp_encounter_data
SET keime_4mrgn = CASE
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:4MRGN:CONF' THEN '4MRGN'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:4MRGN:PB' THEN '4MRGN'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:4MRGN:SUSP' THEN 'V.a. 4MRGN'
    ELSE of.concept_cd
END
FROM i2b2crcdata.observation_fact of
WHERE of.concept_cd LIKE 'AKTIN:PATHOGENE:4MRGN%'
AND of.modifier_cd = '@'
AND of.encounter_num = temp_encounter_data.encounter_num;
-- End Column Keime 4MRGN

-- Column Keime VRE
ALTER TABLE temp_encounter_data ADD COLUMN keime_vre VARCHAR;
UPDATE temp_encounter_data
SET keime_vre = CASE
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:VRE' THEN 'VRE'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:VRE:CONF' THEN 'VRE'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:VRE:PB' THEN 'VRE'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:VRE:SUSP' THEN 'V.a. VRE'
    ELSE of.concept_cd
END
FROM i2b2crcdata.observation_fact of
WHERE of.concept_cd LIKE 'AKTIN:PATHOGENE:VRE%'
AND of.modifier_cd = '@'
AND of.encounter_num = temp_encounter_data.encounter_num;
-- End Column Keime VRE

-- Column Keime Andere MRE
ALTER TABLE temp_encounter_data ADD COLUMN keime_andere VARCHAR;
UPDATE temp_encounter_data
SET keime_andere = CASE
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:OTH' THEN 'Andere MRE'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:OTH:CONF' THEN 'Andere MRE'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:OTH:PB' THEN 'Andere MRE'
    WHEN of.concept_cd = 'AKTIN:PATHOGENE:OTH:SUSP' THEN 'V.a. Andere MRE'
    ELSE of.concept_cd
END
FROM i2b2crcdata.observation_fact of
WHERE of.concept_cd LIKE 'AKTIN:PATHOGENE:OTH%'
AND of.modifier_cd = '@'
AND of.encounter_num = temp_encounter_data.encounter_num;
-- End Column Keime Andere MRE



-- Column Labor
ALTER TABLE temp_encounter_data ADD COLUMN labor VARCHAR;
UPDATE temp_encounter_data SET labor = CASE
    WHEN concept_cd = 'LOINC:26436-6' THEN 'Labor durchgefuehrt'
    WHEN concept_cd = 'LOINC:26436-6:NEG' THEN 'Labor nicht durchgefuehrt'
    ELSE
        concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:26436-6%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Labor

-- Column labor_ts
ALTER TABLE temp_encounter_data ADD COLUMN labor_ts TIMESTAMP;
UPDATE temp_encounter_data SET labor_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:26436-6%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column labor_ts

-- Column BGA
ALTER TABLE temp_encounter_data ADD COLUMN BGA VARCHAR;
UPDATE temp_encounter_data SET BGA = CASE
    WHEN concept_cd = 'LOINC:18767-4' THEN 'BGA durchgefuehrt'
    WHEN concept_cd = 'LOINC:18767-4:NEG' THEN 'BGA nicht durchgefuehrt'
    ELSE
        concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:18767-4%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column BGA

-- Column BGA_ts
ALTER TABLE temp_encounter_data ADD COLUMN BGA_ts TIMESTAMP;
UPDATE temp_encounter_data SET BGA_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:18767-4%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column BGA_ts


-- Column Urinschnelltest
ALTER TABLE temp_encounter_data ADD COLUMN Urinschnelltest VARCHAR;
UPDATE temp_encounter_data SET Urinschnelltest = CASE
    WHEN concept_cd = 'LOINC:50556-0' THEN 'Urinschnelltest durchgefuehrt'
    WHEN concept_cd = 'LOINC:50556-0:NEG' THEN 'Urinschnelltest nicht durchgefuehrt'
    ELSE
        concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:50556-0%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Urinschnelltest

-- Column Urinschnelltest_ts
ALTER TABLE temp_encounter_data ADD COLUMN Urinschnelltest_ts TIMESTAMP;
UPDATE temp_encounter_data SET Urinschnelltest_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:50556-0%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Urinschnelltest_ts

-- Column ekg
ALTER TABLE temp_encounter_data ADD COLUMN ekg VARCHAR;
UPDATE temp_encounter_data SET ekg = CASE
    WHEN concept_cd = 'LOINC:34534-8' THEN 'EKG durchgefuehrt'
    WHEN concept_cd = 'LOINC:34534-8:NEG' THEN 'EKG nicht durchgefuehrt'
    ELSE concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:34534-8%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column ekg

-- Column ekg_ts
ALTER TABLE temp_encounter_data ADD COLUMN ekg_ts TIMESTAMP;
UPDATE temp_encounter_data SET ekg_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:34534-8%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column ekg_ts

-- Column Sonographie
ALTER TABLE temp_encounter_data ADD COLUMN Sonographie VARCHAR;
UPDATE temp_encounter_data SET Sonographie = CASE
    WHEN concept_cd = 'LOINC:25061-3' THEN 'Sonographie durchgefuehrt'
    WHEN concept_cd = 'LOINC:25061-3:NEG' THEN 'Sonographie nicht durchgefuehrt'
    ELSE
        concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:25061-3%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Sonographie

-- Column Sonographie_ts
ALTER TABLE temp_encounter_data ADD COLUMN Sonographie_ts TIMESTAMP;
UPDATE temp_encounter_data SET Sonographie_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:25061-3%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Sonographie_ts

-- Column echo
ALTER TABLE temp_encounter_data ADD COLUMN echo VARCHAR;
UPDATE temp_encounter_data SET echo = CASE
    WHEN concept_cd = 'LOINC:42148-7' THEN 'Echokardiographie durchgefuehrt'
    WHEN concept_cd = 'LOINC:42148-7:NEG' THEN 'Echokardiographie nicht durchgefuehrt'
    ELSE concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:42148-7%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column echo

-- Column echo_ts
ALTER TABLE temp_encounter_data ADD COLUMN echo_ts TIMESTAMP;
UPDATE temp_encounter_data SET echo_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:42148-7%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column echo_ts

-- Column cct
ALTER TABLE temp_encounter_data ADD COLUMN cct VARCHAR;
UPDATE temp_encounter_data SET cct = CASE
    WHEN concept_cd = 'LOINC:24725-4' THEN 'cCT durchgefuehrt'
    WHEN concept_cd = 'LOINC:24725-4:NEG' THEN 'cCT nicht durchgefuehrt'
    ELSE concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:24725-4%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column cct

-- Column cct_ts
ALTER TABLE temp_encounter_data ADD COLUMN cct_ts TIMESTAMP;
UPDATE temp_encounter_data SET cct_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:24725-4%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column cct_ts

-- Column ct
ALTER TABLE temp_encounter_data ADD COLUMN ct VARCHAR;
UPDATE temp_encounter_data SET ct = CASE
    WHEN concept_cd = 'LOINC:25045-6' THEN 'CT durchgefuehrt'
    WHEN concept_cd = 'LOINC:25045-6:NEG' THEN 'CT nicht durchgefuehrt'
    ELSE concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:25045-6%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column ct

-- Column ct_ts
ALTER TABLE temp_encounter_data ADD COLUMN ct_ts TIMESTAMP;
UPDATE temp_encounter_data SET ct_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:25045-6%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column ct_ts

-- Column Traumascan
ALTER TABLE temp_encounter_data ADD COLUMN Traumascan VARCHAR;
UPDATE temp_encounter_data SET Traumascan = CASE
    WHEN concept_cd = 'LOINC:46305-9' THEN 'Traumascan durchgefuehrt'
    WHEN concept_cd = 'LOINC:46305-9:NEG' THEN 'Traumascan nicht durchgefuehrt'
    ELSE
        concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:46305-9%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Traumascan

-- Column Traumascan_ts
ALTER TABLE temp_encounter_data ADD COLUMN Traumascan_ts TIMESTAMP;
UPDATE temp_encounter_data SET Traumascan_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:46305-9%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Traumascan_ts

-- Column Roengten Wirbelsaeule
ALTER TABLE temp_encounter_data ADD COLUMN Roengten_Wirbelsaeule VARCHAR;
UPDATE temp_encounter_data SET Roengten_Wirbelsaeule = CASE
    WHEN concept_cd = 'LOINC:38008-9' THEN 'Roengten Wirbelsaeule durchgefuehrt'
    WHEN concept_cd = 'LOINC:38008-9:NEG' THEN 'Roengten Wirbelsaeule nicht durchgefuehrt'
    ELSE
        concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:38008-9%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Roengten Wirbelsaeule

-- Column Roengten_Wirbelsaeule_ts
ALTER TABLE temp_encounter_data ADD COLUMN Roengten_Wirbelsaeule_ts TIMESTAMP;
UPDATE temp_encounter_data SET Roengten_Wirbelsaeule_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:38008-9%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column Roengten_Wirbelsaeule_ts

-- Column roentgen_thorax
ALTER TABLE temp_encounter_data ADD COLUMN roentgen_thorax VARCHAR;
UPDATE temp_encounter_data SET roentgen_thorax = CASE
    WHEN concept_cd = 'LOINC:30745-4' THEN 'Roengten Thorax durchgefuehrt'
    WHEN concept_cd = 'LOINC:30745-4:NEG' THEN 'Roengten Thorax nicht durchgefuehrt'
    ELSE concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:30745-4%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column roentgen_thorax

-- Column roentgen_thorax_ts
ALTER TABLE temp_encounter_data ADD COLUMN roentgen_thorax_ts TIMESTAMP;
UPDATE temp_encounter_data SET roentgen_thorax_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:30745-4%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column roentgen_thorax_ts

-- Column roentgen_becken
ALTER TABLE temp_encounter_data ADD COLUMN roentgen_becken VARCHAR;
UPDATE temp_encounter_data SET roentgen_becken = CASE
    WHEN concept_cd = 'LOINC:28561-9' THEN 'Roengten Becken durchgefuehrt'
    WHEN concept_cd = 'LOINC:28561-9:NEG' THEN 'Roengten Becken nicht durchgefuehrt'
    ELSE concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:28561-9%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column roentgen_becken

-- Column roentgen_becken_ts
ALTER TABLE temp_encounter_data ADD COLUMN roentgen_becken_ts TIMESTAMP;
UPDATE temp_encounter_data SET roentgen_becken_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:28561-9%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column roentgen_becken_ts

-- Column roentgen_extremitaet
ALTER TABLE temp_encounter_data ADD COLUMN roentgen_extremitaet VARCHAR;
UPDATE temp_encounter_data SET roentgen_extremitaet = CASE
    WHEN concept_cd = 'LOINC:37637-6' THEN 'Roengten Extremität durchgefuehrt'
    WHEN concept_cd = 'LOINC:37637-6:NEG' THEN 'Roengten Extremität nicht durchgefuehrt'
    ELSE concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:37637-6%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column roentgen_extremitaet

-- Column roentgen_extremitaet_ts
ALTER TABLE temp_encounter_data ADD COLUMN roentgen_extremitaet_ts TIMESTAMP;
UPDATE temp_encounter_data SET roentgen_extremitaet_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:37637-6%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column roentgen_extremitaet_ts

-- Column roentgen_sonstiges
ALTER TABLE temp_encounter_data ADD COLUMN roentgen_sonstiges VARCHAR;
UPDATE temp_encounter_data SET roentgen_sonstiges = CASE
    WHEN concept_cd = 'LOINC:43468-8' THEN 'Roengten sonstiges durchgefuehrt'
    WHEN concept_cd = 'LOINC:43468-8:NEG' THEN 'Roengten sonstiges nicht durchgefuehrt'
    ELSE concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:43468-8%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column roentgen_sonstiges

-- Column roentgen_sonstiges_ts
ALTER TABLE temp_encounter_data ADD COLUMN roentgen_sonstiges_ts TIMESTAMP;
UPDATE temp_encounter_data SET roentgen_sonstiges_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:43468-8%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column roentgen_sonstiges_ts

-- Column MRT
ALTER TABLE temp_encounter_data ADD COLUMN mrt VARCHAR;
UPDATE temp_encounter_data SET mrt = CASE
    WHEN concept_cd = 'LOINC:25056-3' THEN 'mrt durchgefuehrt'
    WHEN concept_cd = 'LOINC:25056-3:NEG' THEN 'mrt nicht durchgefuehrt'
    ELSE concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:25056-3%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column MRT

-- Column mrt_ts
ALTER TABLE temp_encounter_data ADD COLUMN mrt_ts TIMESTAMP;
UPDATE temp_encounter_data SET mrt_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'LOINC:25056-3%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column mrt_ts


-- Column verbleib
ALTER TABLE temp_encounter_data ADD COLUMN verbleib VARCHAR;
UPDATE temp_encounter_data SET verbleib  = CASE
    WHEN concept_cd = 'AKTIN:TRANSFER:1' THEN 'Aufnahme in Funktionsbereich'
    WHEN concept_cd = 'AKTIN:TRANSFER:2' THEN 'Verlegung extern in Funktionsbereich'
    WHEN concept_cd = 'AKTIN:TRANSFER:3' THEN 'Aufnahme auf Ueberwachungsstation'
    WHEN concept_cd = 'AKTIN:TRANSFER:4' THEN 'Verlegung extern auf Ueberwachungsstation'
    WHEN concept_cd = 'AKTIN:TRANSFER:5' THEN 'Aufnahme auf Normalstation'
    WHEN concept_cd = 'AKTIN:TRANSFER:6' THEN 'Verlegung extern auf Normalstation'
    WHEN concept_cd = 'AKTIN:DISCHARGE:1' THEN 'Tod'
    WHEN concept_cd = 'AKTIN:DISCHARGE:2' THEN 'Entlassung gegen aerztlichen Rat'
    WHEN concept_cd = 'AKTIN:DISCHARGE:3' THEN 'Behandlung durch Pat. abgebrochen'
    WHEN concept_cd = 'AKTIN:DISCHARGE:4' THEN 'Entlassung nach Hause'
    WHEN concept_cd = 'AKTIN:DISCHARGE:5' THEN 'Entlassung zu weiterbehandelnden Arzt'
    WHEN concept_cd = 'AKTIN:DISCHARGE:6' THEN 'kein Arztkontakt'
    WHEN concept_cd = 'AKTIN:DISCHARGE:OTH' THEN 'Sonstige Entlassung'
    ELSE
        concept_cd
    END
FROM i2b2crcdata.observation_fact
WHERE ((concept_cd LIKE '%TRANSFER%' AND concept_cd <> 'AKTIN:TRANSFER:ZeitpunktVerlegung') OR concept_cd LIKE '%DISCHARGE%') AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column verbleib

-- Column triage_ts
ALTER TABLE temp_encounter_data ADD COLUMN triage_ts TIMESTAMP;
UPDATE temp_encounter_data SET triage_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd LIKE 'MTS%' OR concept_cd LIKE 'ESI%' OR concept_cd LIKE 'AKTIN:ASSESSMENT') AND modifier_cd = 'effectiveTimeLow'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column triage_ts

-- Column start_arztkontakt_ts
ALTER TABLE temp_encounter_data ADD COLUMN start_arztkontakt_ts TIMESTAMP;
UPDATE temp_encounter_data SET start_arztkontakt_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd = 'AKTIN:PHYSENCOUNTER' OR concept_cd = 'AKTIN:ZeitpunktErsterArztkontakt') AND modifier_cd = 'timeLow'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column start_arztkontakt_ts

-- Column start_therapie_ts
ALTER TABLE temp_encounter_data ADD COLUMN start_therapie_ts TIMESTAMP;
UPDATE temp_encounter_data SET start_therapie_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE (concept_cd = 'AKTIN:STARTTHERAPY' OR concept_cd = 'AKTIN:ZeitpunktTherapiebeginn')
AND modifier_cd = 'effectiveTimeLow'
AND i2b2crcdata.observation_fact.encounter_num = temp_encounter_data.encounter_num;
-- End Column start_therapie_ts


-- Column aktin_diagnose_table
-- ### Table 2 : Patient diagnoses (Datensatz Notaufnahme) ###
CREATE TEMPORARY TABLE temp_diagnoses AS
SELECT DISTINCT
temp_joindata.encounter_num,
instance_num,
substr(i2b2crcdata.observation_fact.concept_cd, 9) as icd_code
FROM temp_joindata
INNER JOIN i2b2crcdata.observation_fact
ON temp_joindata.encounter_num = i2b2crcdata.observation_fact.encounter_num
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'ICD10GM:%' AND i2b2crcdata.observation_fact.provider_id = '@' and i2b2crcdata.observation_fact.modifier_cd = '@';


-- Markierung der führenden Notaufnahmediagnose
ALTER TABLE temp_diagnoses ADD COLUMN diagnose_fuehrend VARCHAR;
UPDATE temp_diagnoses SET diagnose_fuehrend = 'f'
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'ICD10GM:%' AND provider_id = '@' AND modifier_cd = 'AKTIN:DIAG:F'
AND i2b2crcdata.observation_fact.encounter_num = temp_diagnoses.encounter_num
AND i2b2crcdata.observation_fact.instance_num = temp_diagnoses.instance_num
AND substr(i2b2crcdata.observation_fact.concept_cd, 9) = temp_diagnoses.icd_code;


-- Zusatzkennzeichen (zur Diagnosesicherheit)
ALTER TABLE temp_diagnoses ADD COLUMN diagnose_zusatz VARCHAR;
UPDATE temp_diagnoses SET diagnose_zusatz = substr(modifier_cd, 12)
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'ICD10GM:%'
AND provider_id = '@'
AND (modifier_cd LIKE 'AKTIN:DIAG:%' AND modifier_cd NOT LIKE 'AKTIN:DIAG:F')
AND i2b2crcdata.observation_fact.encounter_num = temp_diagnoses.encounter_num
AND i2b2crcdata.observation_fact.instance_num = temp_diagnoses.instance_num
AND substr(i2b2crcdata.observation_fact.concept_cd, 9) = temp_diagnoses.icd_code;

--delete empty cases and instance_num
ALTER TABLE temp_diagnoses DROP COLUMN instance_num;
DELETE FROM temp_diagnoses WHERE icd_code is NULL;
-- End Column aktin_diagnose_table


-- Column ICD_table
-- ### Table 3 : Patient diagnoses (P21) ###
CREATE TEMPORARY TABLE temp_icd AS
SELECT
temp_joindata.encounter_num,
instance_num,
substr(i2b2crcdata.observation_fact.concept_cd, 9) as icd_code
FROM temp_joindata
INNER JOIN i2b2crcdata.observation_fact
ON temp_joindata.encounter_num = i2b2crcdata.observation_fact.encounter_num
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'ICD10GM:%' AND i2b2crcdata.observation_fact.provider_id = 'P21' and i2b2crcdata.observation_fact.modifier_cd = '@';


-- Diagnoseart
ALTER TABLE temp_icd ADD COLUMN diagnoseart VARCHAR;
UPDATE temp_icd SET diagnoseart = tval_char
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'ICD10GM:%' AND provider_id = 'P21' AND modifier_cd = 'diagType'
AND i2b2crcdata.observation_fact.encounter_num = temp_icd.encounter_num
AND i2b2crcdata.observation_fact.instance_num = temp_icd.instance_num
AND substr(concept_cd, 9) = temp_icd.icd_code;

-- ICD-zugehörige HD
ALTER TABLE temp_icd ADD COLUMN zugehörige_HD VARCHAR;
UPDATE temp_icd SET zugehörige_HD = substr(tval_char,9)
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'ICD10GM:%'  AND provider_id = 'P21' AND modifier_cd = 'sdFrom'
AND i2b2crcdata.observation_fact.encounter_num = temp_icd.encounter_num
AND i2b2crcdata.observation_fact.instance_num = temp_icd.instance_num
AND substr(concept_cd, 9) = temp_icd.icd_code;


-- ICD-Version
ALTER TABLE temp_icd ADD COLUMN icd_version VARCHAR;
UPDATE temp_icd SET icd_version = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'ICD10GM:%'  AND provider_id = 'P21' AND modifier_cd = 'cdVersion'
AND i2b2crcdata.observation_fact.encounter_num = temp_icd.encounter_num
AND i2b2crcdata.observation_fact.instance_num = temp_icd.instance_num
AND substr(concept_cd, 9) = temp_icd.icd_code;

-- ICD-Diagnosesicherheit
ALTER TABLE temp_icd ADD COLUMN icd_diagnosesicherheit VARCHAR;
UPDATE temp_icd SET icd_diagnosesicherheit = tval_char
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'ICD10GM:%'
AND provider_id = 'P21'
AND modifier_cd = 'certainty'
AND i2b2crcdata.observation_fact.encounter_num = temp_icd.encounter_num
AND i2b2crcdata.observation_fact.instance_num = temp_icd.instance_num
AND substr(concept_cd, 9) = temp_icd.icd_code;

-- ICD-lokalisierung
ALTER TABLE temp_icd ADD COLUMN icd_lokalisierung VARCHAR;
UPDATE temp_icd SET icd_lokalisierung = tval_char
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'ICD10GM:%'  AND provider_id = 'P21' AND modifier_cd = 'localisation'
AND i2b2crcdata.observation_fact.encounter_num = temp_icd.encounter_num
AND i2b2crcdata.observation_fact.instance_num = temp_icd.instance_num
AND substr(concept_cd, 9) = temp_icd.icd_code;

ALTER TABLE temp_icd ADD COLUMN sekundär_Diagnosesicherheit VARCHAR;
UPDATE temp_icd SET sekundär_Diagnosesicherheit = icd_diagnosesicherheit
WHERE zugehörige_HD IS NOT NULL AND zugehörige_HD <> '';

ALTER TABLE temp_icd ADD COLUMN sekundär_lokalisierung VARCHAR;
UPDATE temp_icd SET sekundär_lokalisierung = icd_lokalisierung
WHERE zugehörige_HD IS NOT NULL AND zugehörige_HD <> '';

--delete empty cases and instance_num
ALTER TABLE temp_icd DROP COLUMN instance_num;
delete from temp_icd WHERE icd_code is NULL;
-- End Column ICD_table


-- ### Table 4: Patient Fall (P21) ###
-- Fallnummer
-- Merge over Discharge reason which is mandatory item
CREATE TEMPORARY TABLE temp_fall AS
SELECT DISTINCT
encounter_num
FROM temp_joindata;

-- Column FALL_Versicherungsnummer
ALTER TABLE temp_fall ADD COLUMN iknr VARCHAR;
UPDATE temp_fall SET iknr = i2b2crcdata.observation_fact.tval_char
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd = 'AKTIN:IKNR' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Versicherungsnummer

-- Column FALL_Geburtsjahr
ALTER TABLE temp_fall ADD COLUMN geburtsjahr NUMERIC;
UPDATE temp_fall SET geburtsjahr = i2b2crcdata.observation_fact.nval_num
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd = 'LOINC:80904-6' AND provider_id = 'P21' AND modifier_cd = '@'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Geburtsjahr

-- Column FALL_Geschlecht
ALTER TABLE temp_fall ADD COLUMN geschlecht VARCHAR;
UPDATE temp_fall SET geschlecht = SUBSTR(i2b2crcdata.observation_fact.concept_cd, 9)
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'P21:SEX%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Geschlecht

-- Column FALL_zipcode
ALTER TABLE temp_fall ADD COLUMN zipcode VARCHAR;
UPDATE temp_fall SET zipcode = i2b2crcdata.observation_fact.tval_char
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd = 'AKTIN:ZIPCODE' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_zipcode

-- Column FALL_aufnahmedatum
ALTER TABLE temp_fall ADD COLUMN aufnahmedatum TIMESTAMP;
UPDATE temp_fall SET aufnahmedatum = start_date
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'P21:ADMR%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_aufnahmedatum

-- Column FALL_Aufnahmegrund
ALTER TABLE temp_fall ADD COLUMN aufnahmegrund VARCHAR;
UPDATE temp_fall SET aufnahmegrund = SUBSTR(i2b2crcdata.observation_fact.concept_cd, 10)
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'P21:ADMR%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Aufnahmegrund

-- Column FALL_Aufnahmeanlass
ALTER TABLE temp_fall ADD COLUMN aufnahmeanlass VARCHAR;
UPDATE temp_fall SET aufnahmeanlass = SUBSTR(i2b2crcdata.observation_fact.concept_cd, 10)
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'P21:ADMC%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Aufnahmeanlass


-- Column FALL_Fallzusammenführung_Grund
ALTER TABLE temp_fall ADD COLUMN fallzusammenführung_grund VARCHAR;
UPDATE temp_fall SET fallzusammenführung_grund = SUBSTR(i2b2crcdata.observation_fact.concept_cd, 11)
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'P21:MERGE%'
AND i2b2crcdata.observation_fact.concept_cd NOT LIKE 'P21:MERGE:J%'
AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Fallzusammenführung_Grund

-- Column FALL_Fallzusammenführung
ALTER TABLE temp_fall ADD COLUMN fallzusammenführung VARCHAR;
UPDATE temp_fall SET fallzusammenführung = CASE
    WHEN SUBSTR(concept_cd, 11) = 'J' THEN 'Y' ELSE 'NA'
END
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'P21:MERGE:J%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Fallzusammenführung

-- Column Fall_Verweildauer_Intensiv
ALTER TABLE temp_fall ADD COLUMN verweildauer_intensiv NUMERIC;
UPDATE temp_fall SET Verweildauer_Intensiv = nval_num
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'P21:DCC%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column Fall_Verweildauer_Intensiv

-- Column FALL_entlassung_ts
ALTER TABLE temp_fall ADD COLUMN entlassung_ts TIMESTAMP;
UPDATE temp_fall SET entlassung_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'P21:DISR%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_entlassung_ts

-- Column FALL_Entlassgrund
ALTER TABLE temp_fall ADD COLUMN entlassungsgrund VARCHAR;
UPDATE temp_fall SET entlassungsgrund = SUBSTR(i2b2crcdata.observation_fact.concept_cd, 10)
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'P21:DISR%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Entlassgrund

-- Column FALL_Beatmungsstunden
ALTER TABLE temp_fall ADD COLUMN beatmungsstunden NUMERIC;
UPDATE temp_fall SET beatmungsstunden = nval_num
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'P21:DV%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Beatmungsstunden

-- Column FALL_Behandlungsbeginn_vorstationär_ts
ALTER TABLE temp_fall ADD COLUMN behandlungsbeginn_vorstationär_ts TIMESTAMP;
UPDATE temp_fall SET behandlungsbeginn_vorstationär_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'P21:PREADM%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Behandlungsbeginn_vorstationär_ts

-- Column FALL_Behandlungstage_vorstationär
ALTER TABLE temp_fall ADD COLUMN behandlungstage_vorstationär NUMERIC;
UPDATE temp_fall SET behandlungstage_vorstationär = nval_num
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'P21:PREADM%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Behandlungstage_vorstationär


-- Column FALL_Behandlungsende_nachstationär_ts
ALTER TABLE temp_fall ADD COLUMN behandlungsende_nachstationär_ts TIMESTAMP;
UPDATE temp_fall SET behandlungsende_nachstationär_ts = start_date
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'P21:POSTDIS%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Behandlungsende_nachstationär_ts

-- Column FALL_Behandlungstage_nachstationär
ALTER TABLE temp_fall ADD COLUMN behandlungstage_nachstationär NUMERIC;
UPDATE temp_fall SET behandlungstage_nachstationär = nval_num
FROM i2b2crcdata.observation_fact
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'P21:POSTDIS%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fall.encounter_num;
-- End Column FALL_Behandlungstage_nachstationär

--delete empty cases
delete from temp_fall WHERE aufnahmedatum is NULL;



-- Column FAB_table
-- ### Table 5: Fachabteilung (P21) ###
CREATE TEMPORARY TABLE temp_fab AS
SELECT
temp_joindata.encounter_num,
instance_num,
tval_char as fab_fachabteilung
FROM temp_joindata
INNER JOIN i2b2crcdata.observation_fact
ON temp_joindata.encounter_num = i2b2crcdata.observation_fact.encounter_num
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'P21:DEP%' AND i2b2crcdata.observation_fact.provider_id = 'P21' and i2b2crcdata.observation_fact.modifier_cd = '@';

-- FAB_aufnahmedatum
ALTER TABLE temp_fab ADD COLUMN fab_aufnahmedatum TIMESTAMP;
UPDATE temp_fab SET fab_aufnahmedatum = start_date
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'P21:DEP%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fab.encounter_num
AND i2b2crcdata.observation_fact.instance_num = temp_fab.instance_num;

-- FAB_entlassdatum
ALTER TABLE temp_fab ADD COLUMN fab_entlassungsdatum TIMESTAMP;
UPDATE temp_fab SET fab_entlassungsdatum = end_date
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'P21:DEP%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fab.encounter_num
AND i2b2crcdata.observation_fact.instance_num = temp_fab.instance_num;

-- FAB_intensivbett
ALTER TABLE temp_fab ADD COLUMN intensivbett VARCHAR;
UPDATE temp_fab SET intensivbett = CASE
    WHEN SUBSTR(concept_cd, 9) = 'CC' THEN 'Y' ELSE 'N'
END
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'P21:DEP%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.encounter_num = temp_fab.encounter_num
AND i2b2crcdata.observation_fact.instance_num = temp_fab.instance_num;

--delete empty cases
ALTER TABLE temp_fab DROP COLUMN instance_num;
delete from temp_fab WHERE fab_aufnahmedatum is NULL;
-- End Column FAB_table

-- Column OPS_table
-- ### Table 6: Table with P21 OPS data ###
CREATE TEMPORARY TABLE temp_ops AS
SELECT
temp_joindata.encounter_num,
instance_num,
SUBSTR(concept_cd, 5) as ops_code
FROM temp_joindata
INNER JOIN i2b2crcdata.observation_fact
ON temp_joindata.encounter_num = i2b2crcdata.observation_fact.encounter_num
WHERE i2b2crcdata.observation_fact.concept_cd LIKE 'OPS%' AND i2b2crcdata.observation_fact.provider_id = 'P21' and i2b2crcdata.observation_fact.modifier_cd = '@';


-- OPS_datum
ALTER TABLE temp_ops ADD COLUMN ops_datum TIMESTAMP;
UPDATE temp_ops SET ops_datum = start_date
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'OPS%' AND provider_id = 'P21'
AND i2b2crcdata.observation_fact.instance_num = temp_ops.instance_num
AND i2b2crcdata.observation_fact.encounter_num = temp_ops.encounter_num;
-- End Column OPS_datum

-- ops_version
ALTER TABLE temp_ops ADD COLUMN ops_version NUMERIC;
UPDATE temp_ops SET ops_version = nval_num
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'OPS%'
AND provider_id = 'P21'
AND modifier_cd = 'cdVersion'
AND i2b2crcdata.observation_fact.instance_num = temp_ops.instance_num
AND i2b2crcdata.observation_fact.encounter_num = temp_ops.encounter_num;
-- End column ops_version

-- ops_lokalisation
ALTER TABLE temp_ops ADD COLUMN ops_lokalisation VARCHAR;
UPDATE temp_ops
SET ops_lokalisation = observation_fact.tval_char
FROM i2b2crcdata.observation_fact
WHERE concept_cd LIKE 'OPS:%' AND provider_id = 'P21' AND modifier_cd = 'localisation'
AND i2b2crcdata.observation_fact.encounter_num = temp_ops.encounter_num
AND i2b2crcdata.observation_fact.instance_num = temp_ops.instance_num
AND substr(concept_cd, 5) = temp_ops.ops_code;
-- End column ops_lokalisation

--delete empty cases
ALTER TABLE temp_ops DROP COLUMN instance_num;
delete from temp_ops WHERE ops_code is NULL;
-- End Column OPS_table

SELECT temp_encounter_data.altersgruppe as age,
       temp_encounter_data.triage as triage_score,
       aufnahme_ts as admission_time,
       temp_encounter_data.cedis as cedis_code
       FROM  temp_encounter_data
--         LIMIT 1;
