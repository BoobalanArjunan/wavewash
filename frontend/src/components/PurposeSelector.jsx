import { Mic, Speaker, Users, Music4, SlidersHorizontal } from 'lucide-react'

const ICONS = {
  asr: Mic,
  tts: Speaker,
  speaker_id: Users,
  music_gen: Music4,
  custom: SlidersHorizontal,
}

export default function PurposeSelector({ purposes, selected, onSelect }) {
  return (
    <div className="grid grid-cols-1 gap-2">
      {purposes.map((p) => {
        const Icon = ICONS[p.key] || SlidersHorizontal
        const isSelected = selected === p.key
        return (
          <button
            key={p.key}
            onClick={() => onSelect(p.key)}
            className={`flex items-start gap-3 p-3 rounded-lg border text-left transition-all ${
              isSelected
                ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10'
                : 'border-[var(--color-border)] bg-[var(--color-panel)] hover:border-[var(--color-text-faint)]'
            }`}
          >
            <div className={`p-2 rounded-md shrink-0 ${isSelected ? 'bg-[var(--color-accent)] text-[#14161A]' : 'bg-[var(--color-panel-raised)] text-[var(--color-text-muted)]'}`}>
              <Icon size={16} />
            </div>
            <div className="min-w-0">
              <div className="text-sm font-semibold">{p.name}</div>
              <div className="text-xs text-[var(--color-text-muted)] mt-0.5 leading-snug">{p.description}</div>
            </div>
          </button>
        )
      })}
    </div>
  )
}
