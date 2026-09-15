# Lesson 09: Observer Engineering Report

## Observability
- rank: 2 / 2
- condition_number: 2004.06
- sigma_min: 0.000705673

## Nominal observer poles
- requested: [0.9326339+0.j 0.6710479+0.j]
- achieved: [0.6710479 0.9326339]

## Findings

1. Increasing observer speed reduces initial-state convergence time but can amplify encoder noise into velocity and torque.
2. The observer uses the applied torque after saturation; prediction with requested torque would violate plant-input consistency.
3. A constant encoder bias is not a state in this model, so innovation and estimated position can remain biased; use bias augmentation or a Kalman-style estimator later.

## How to interpret these results

- The model is observable because rank is 2, but condition number 2004.1 and sigma_min 0.000706 show that velocity information is numerically weak at a 1 ms interval.
- Observer 2x gives velocity RMSE 0.553 rad/s, while 10x reaches 1.472 rad/s. Faster poles improve position correction but inject more encoder noise into velocity and torque.
- A late or NaN convergence time in noisy runs means the estimate did not remain inside the strict 2% error band; it does not mean that the observer poles are unstable.
- Raw difference produces 2.077 Nm torque RMS and 27.35% saturation. The configured LPF gives the best velocity RMSE in this simple plant, so a pole-placed observer is not automatically the best noise filter.
- Ten-times encoder noise raises innovation RMS from 0.0071 to 0.0121 rad and torque RMS from 0.341 to 0.719 Nm.
- A 0.5 degree encoder offset produces 0.568 degree position-estimate RMSE and no 2% convergence, because bias is absent from the observer state.

The full-run RMSE includes the deliberately large initial mismatch q=20 deg and q_hat=0 deg. Compare convergence time and steady curves in the figures before selecting an observer speed.

## Enterprise deployment guidance

1. Initialize estimated position from the first validated encoder sample and start estimated velocity conservatively.
2. Tune observer bandwidth offline from recorded q_measured and u_applied; increase speed only while innovation, velocity noise, and torque remain acceptable.
3. Feed actuator-applied torque after current and torque limits into the observer model.
4. Treat persistent innovation mean, periodic structure, and spikes as diagnostics for bias, missing dynamics, disturbances, or encoder faults.
5. Use bias-state augmentation and Kalman/EKF sensor fusion when payload variation, process noise, or multiple sensors dominate.

## Metrics

### convergence
- q_est_rmse_rad: 0.00444413
- dq_est_rmse_rad_s: 0.861739
- observer_convergence_time_s: 0.079
- innovation_rms_rad: 0.00708604
- tracking_rmse_rad: 0.0307039
- rms_torque_nm: 0.335723
- peak_torque_nm: 3
- saturation_ratio_percent: 1
- control_velocity_rmse_rad_s: 0.869256
- control_velocity_error_std_rad_s: 0.865656

### speed_tradeoff
#### observer_2x
- q_est_rmse_rad: 0.00716077
- dq_est_rmse_rad_s: 0.548511
- observer_convergence_time_s: 2.881
- innovation_rms_rad: 0.00907149
- tracking_rmse_rad: 0.0328013
- rms_torque_nm: 0.292799
- peak_torque_nm: 3
- saturation_ratio_percent: 0.525
- control_velocity_rmse_rad_s: 0.552947
- control_velocity_error_std_rad_s: 0.548345
#### observer_4x
- q_est_rmse_rad: 0.00447722
- dq_est_rmse_rad_s: 0.862172
- observer_convergence_time_s: 0.082
- innovation_rms_rad: 0.00714795
- tracking_rmse_rad: 0.0307233
- rms_torque_nm: 0.3412
- peak_torque_nm: 3
- saturation_ratio_percent: 0.975
- control_velocity_rmse_rad_s: 0.869764
- control_velocity_error_std_rad_s: 0.866166
#### observer_10x
- q_est_rmse_rad: 0.00223315
- dq_est_rmse_rad_s: 1.46367
- observer_convergence_time_s: 3.976
- innovation_rms_rad: 0.00600499
- tracking_rmse_rad: 0.0244742
- rms_torque_nm: 0.374003
- peak_torque_nm: 3
- saturation_ratio_percent: 0.8
- control_velocity_rmse_rad_s: 1.47189
- control_velocity_error_std_rad_s: 1.4695

### velocity_sources
#### raw_difference
- q_est_rmse_rad: 0.00447846
- dq_est_rmse_rad_s: 0.864741
- observer_convergence_time_s: 0.077
- innovation_rms_rad: 0.00714852
- tracking_rmse_rad: 0.0170368
- rms_torque_nm: 2.07737
- peak_torque_nm: 3
- saturation_ratio_percent: 27.35
- control_velocity_rmse_rad_s: 1.3264
- control_velocity_error_std_rad_s: 1.3264
#### filtered_difference
- q_est_rmse_rad: 0.00447803
- dq_est_rmse_rad_s: 0.86514
- observer_convergence_time_s: 0.077
- innovation_rms_rad: 0.00714836
- tracking_rmse_rad: 0.0162691
- rms_torque_nm: 0.306058
- peak_torque_nm: 3
- saturation_ratio_percent: 0.275
- control_velocity_rmse_rad_s: 0.132419
- control_velocity_error_std_rad_s: 0.132419
#### observer
- q_est_rmse_rad: 0.00447722
- dq_est_rmse_rad_s: 0.862172
- observer_convergence_time_s: 0.082
- innovation_rms_rad: 0.00714795
- tracking_rmse_rad: 0.0307233
- rms_torque_nm: 0.3412
- peak_torque_nm: 3
- saturation_ratio_percent: 0.975
- control_velocity_rmse_rad_s: 0.869764
- control_velocity_error_std_rad_s: 0.866166

### payload
- q_est_rmse_rad: 0.00445784
- dq_est_rmse_rad_s: 0.847127
- observer_convergence_time_s: 0.19
- innovation_rms_rad: 0.00713577
- tracking_rmse_rad: 0.0303406
- rms_torque_nm: 0.342103
- peak_torque_nm: 3
- saturation_ratio_percent: 0.9
- control_velocity_rmse_rad_s: 0.852992
- control_velocity_error_std_rad_s: 0.849383

### bias
- q_est_rmse_rad: 0.00990624
- dq_est_rmse_rad_s: 0.88429
- observer_convergence_time_s: nan
- innovation_rms_rad: 0.00732354
- tracking_rmse_rad: 0.0327465
- rms_torque_nm: 0.34272
- peak_torque_nm: 3
- saturation_ratio_percent: 1.025
- control_velocity_rmse_rad_s: 0.892081
- control_velocity_error_std_rad_s: 0.888391

### normal
- q_est_rmse_rad: 0.00447722
- dq_est_rmse_rad_s: 0.862172
- observer_convergence_time_s: 0.082
- innovation_rms_rad: 0.00714795
- tracking_rmse_rad: 0.0307233
- rms_torque_nm: 0.3412
- peak_torque_nm: 3
- saturation_ratio_percent: 0.975
- control_velocity_rmse_rad_s: 0.869764
- control_velocity_error_std_rad_s: 0.866166

### disturbance
- q_est_rmse_rad: 0.00445784
- dq_est_rmse_rad_s: 0.847127
- observer_convergence_time_s: 0.19
- innovation_rms_rad: 0.00713577
- tracking_rmse_rad: 0.0303406
- rms_torque_nm: 0.342103
- peak_torque_nm: 3
- saturation_ratio_percent: 0.9
- control_velocity_rmse_rad_s: 0.852992
- control_velocity_error_std_rad_s: 0.849383

### stress
- q_est_rmse_rad: 0.00649508
- dq_est_rmse_rad_s: 0.894806
- observer_convergence_time_s: nan
- innovation_rms_rad: 0.012129
- tracking_rmse_rad: 0.0308993
- rms_torque_nm: 0.718618
- peak_torque_nm: 3
- saturation_ratio_percent: 0.925
- control_velocity_rmse_rad_s: 0.910422
- control_velocity_error_std_rad_s: 0.906985
