
DEFAULT_COLUMNS = ('CPU Temperature [°C]', 'GPU Temperature [°C]', 'Memory Temperature [°C]', 'Hot Spot [°C]',
                   'GPU Chip Power Draw [W]', 'GPU Load [%]', 'Fan 1 Speed (%) [%]', 'System Memory Used [MB]')

PERF_CAP_REASON = {
    # Taken from:
    #   https://www.techpowerup.com/forums/threads/gpu-z-perfcap-log-number-meanings.202433/
    1: "NV_GPU_PERF_POLICY_ID_SW_POWER",        # Power. Indicating perf is limited by total power limit.
    2: "NV_GPU_PERF_POLICY_ID_SW_THERMAL",      # Thermal. Indicating perf is limited by temperature limit.
    4: "NV_GPU_PERF_POLICY_ID_SW_RELIABILITY",  # Reliability. Indicating perf is limited by reliability voltage.
    8: "NV_GPU_PERF_POLICY_ID_SW_OPERATING",    # Operating. Indicating perf is limited by max operating voltage.
    16: "NV_GPU_PERF_POLICY_ID_SW_UTILIZATION"  # Utilization. Indicating perf is limited by GPU utilization.
}
