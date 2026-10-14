/**
 * presets.js
 * -----------
 * Translates plain-language choices ("Strict" / "Balanced" / "Lenient", and a
 * named loudness target) into the actual threshold overrides the backend
 * understands. This is what lets someone who has never heard of "SNR" or
 * "LUFS" still configure a real, meaningful run.
 *
 * Honesty note: only the loudness targets below are real external standards
 * (cited). The three strictness levels are reasonable engineering choices we
 * made, scaled from each purpose's own default profile -- they are not
 * claiming to be an industry-codified standard, and the UI should not imply
 * they are.
 */

export const LOUDNESS_TARGETS = [
  {
    key: 'default',
    label: "This purpose's default",
    value: null, // uses whatever the profile already has
    note: null,
  },
  {
    key: 'broadcast',
    label: 'Broadcast (-23 LUFS)',
    value: -23.0,
    note: 'EBU R128 / ITU-R BS.1770 broadcast standard.',
  },
  {
    key: 'streaming',
    label: 'Streaming / music (-14 LUFS)',
    value: -14.0,
    note: 'Matches Spotify, YouTube, and Amazon Music’s normalization target.',
  },
  {
    key: 'podcast',
    label: 'Podcast (-16 LUFS)',
    value: -16.0,
    note: 'Matches Apple’s Sound Check target, commonly used for spoken-word/podcast delivery.',
  },
]

// Multipliers applied to a purpose's own default profile values.
// "balanced" = 1.0 everywhere = the profile's own defaults, unchanged.
const STRICTNESS_MULTIPLIERS = {
  strict: {
    min_snr_db: 1.3,
    max_clipped_ratio: 0.5,
    max_silence_ratio: 0.85,
    loudness_tolerance_db: 0.6,
    declip_max_run_ms: 0.5,
    lossy_cliff_db_per_octave: 0.8,
    forceChecks: { enable_reverb_check: true, enable_rolloff_check: true, enable_crest_factor_check: true },
  },
  balanced: {
    min_snr_db: 1.0,
    max_clipped_ratio: 1.0,
    max_silence_ratio: 1.0,
    loudness_tolerance_db: 1.0,
    declip_max_run_ms: 1.0,
    lossy_cliff_db_per_octave: 1.0,
    forceChecks: {},
  },
  lenient: {
    min_snr_db: 0.55,
    max_clipped_ratio: 3.0,
    max_silence_ratio: 1.08,
    loudness_tolerance_db: 2.0,
    declip_max_run_ms: 2.5,
    lossy_cliff_db_per_octave: 1.6,
    forceChecks: { enable_reverb_check: false, enable_crest_factor_check: false },
  },
}

export const STRICTNESS_LEVELS = [
  {
    key: 'strict',
    label: 'Strict',
    tagline: 'Only near-perfect audio passes',
    description: 'Rejects anything with noticeable noise, clipping, or silence. Use this when every file needs to sound clean on its own — premium production, client-facing voice data, reference music.',
  },
  {
    key: 'balanced',
    label: 'Balanced',
    tagline: 'Recommended default',
    description: 'Matches common practice for preparing ML training data — catches real problems, auto-fixes what it reasonably can, and doesn’t throw away usable audio over minor imperfections.',
  },
  {
    key: 'lenient',
    label: 'Lenient',
    tagline: 'Keep as much data as possible',
    description: 'Only rejects audio that’s genuinely broken (heavy distortion, near-total silence, corrupt files). Good for an early pass on a messy dataset, or when you have limited source material.',
  },
]

function clampRatio(v) {
  return Math.max(0, Math.min(0.99, v))
}

/**
 * Given a purpose's base profile and the chosen strictness + loudness target,
 * returns the override object to send to the backend.
 */
export function buildPresetOverrides(profile, strictnessKey, loudnessKey) {
  if (!profile) return {}
  const mult = STRICTNESS_MULTIPLIERS[strictnessKey] || STRICTNESS_MULTIPLIERS.balanced
  const overrides = {}

  if (profile.min_snr_db != null) overrides.min_snr_db = +(profile.min_snr_db * mult.min_snr_db).toFixed(2)
  if (profile.max_clipped_ratio != null) overrides.max_clipped_ratio = clampRatio(profile.max_clipped_ratio * mult.max_clipped_ratio)
  if (profile.max_silence_ratio != null) overrides.max_silence_ratio = clampRatio(profile.max_silence_ratio * mult.max_silence_ratio)
  if (profile.loudness_tolerance_db != null) overrides.loudness_tolerance_db = +(profile.loudness_tolerance_db * mult.loudness_tolerance_db).toFixed(2)
  if (profile.declip_max_run_ms != null) overrides.declip_max_run_ms = +(profile.declip_max_run_ms * mult.declip_max_run_ms).toFixed(1)

  // How steep a high-frequency drop-off must be before a file is called "compressed/filtered".
  // Higher = fewer files flagged. (Natural instrument roll-off measures 9-22 dB/oct; filters/codecs 49+.)
  if (profile.lossy_cliff_db_per_octave != null) {
    overrides.lossy_cliff_db_per_octave = Math.round(profile.lossy_cliff_db_per_octave * mult.lossy_cliff_db_per_octave)
  }

  Object.assign(overrides, mult.forceChecks)

  const loudness = LOUDNESS_TARGETS.find(l => l.key === loudnessKey)
  if (loudness && loudness.value != null) {
    overrides.target_lufs = loudness.value
  }

  return overrides
}
