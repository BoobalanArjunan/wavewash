import { ShieldCheck, ChevronDown, Info } from 'lucide-react'
import { STRICTNESS_LEVELS, LOUDNESS_TARGETS } from '../presets'
import { useState } from 'react'

export default function QualityPresets({ strictness, onStrictnessChange, loudness, onLoudnessChange }) {
  const [showLoudness, setShowLoudness] = useState(false)

  return (
    <div className="space-y-3">
      <div className="space-y-2">
        {STRICTNESS_LEVELS.map((level) => {
          const isSelected = strictness === level.key
          return (
            <button
              key={level.key}
              onClick={() => onStrictnessChange(level.key)}
              className={`w-full text-left p-3 rounded-lg border transition-all ${
                isSelected
                  ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10'
                  : 'border-[var(--color-border)] bg-[var(--color-panel)] hover:border-[var(--color-text-faint)]'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-semibold">{level.label}</span>
                {level.key === 'balanced' && (
                  <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-[var(--color-accent)]/20 text-[var(--color-accent)]">
                    {level.tagline}
                  </span>
                )}
              </div>
              <p className="text-xs text-[var(--color-text-muted)] leading-snug">{level.description}</p>
            </button>
          )
        })}
      </div>

      <div className="border border-[var(--color-border)] rounded-lg overflow-hidden">
        <button
          onClick={() => setShowLoudness(!showLoudness)}
          className="w-full flex items-center justify-between px-3 py-2 bg-[var(--color-panel)] hover:bg-[var(--color-panel-raised)] transition-colors"
        >
          <span className="text-xs font-medium text-[var(--color-text-muted)]">Loudness target (optional)</span>
          <ChevronDown size={13} className={`text-[var(--color-text-faint)] transition-transform ${showLoudness ? 'rotate-180' : ''}`} />
        </button>
        {showLoudness && (
          <div className="p-3 space-y-1.5 bg-[var(--color-bg)]">
            {LOUDNESS_TARGETS.map((t) => (
              <label key={t.key} className="flex items-start gap-2 cursor-pointer py-1">
                <input
                  type="radio"
                  name="loudness"
                  checked={loudness === t.key}
                  onChange={() => onLoudnessChange(t.key)}
                  className="accent-[var(--color-accent)] mt-0.5"
                />
                <div>
                  <div className="text-xs">{t.label}</div>
                  {t.note && <div className="text-[10px] text-[var(--color-text-faint)]">{t.note}</div>}
                </div>
              </label>
            ))}
          </div>
        )}
      </div>

      <div className="flex items-start gap-2 px-1 pt-1">
        <Info size={12} className="text-[var(--color-text-faint)] mt-0.5 shrink-0" />
        <p className="text-[10px] text-[var(--color-text-faint)] leading-snug">
          These presets adjust the same underlying thresholds an engineer would set by hand.
          Switch to Advanced mode below to see or fine-tune the exact numbers.
        </p>
      </div>
    </div>
  )
}
