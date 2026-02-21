{
    for (i = 1; i <= NF; ++i) {
        if ($i ~ /Age|gender|ethnicity|smoking_status|alcohol_consumption_per_week|physical_activity_minutes_per_week|diet_score|sleep_hours_per_day|screen_time_hours_per_day|family_history_diabetes|hypertension_history|cardiovascular_history|bmi|waist_to_hip_ratio|diabetes_stage/) {
            print i
        }
    }
}