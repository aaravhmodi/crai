import React from 'react'
import { cn } from '@/lib/utils'

export interface StatusIndicatorProps {
  status: 'idle' | 'recording' | 'cry-detected' | 'analyzing' | 'results'
  className?: string
}

const StatusIndicator = React.forwardRef<
  HTMLDivElement,
  StatusIndicatorProps
>(({ status, className, ...props }, ref) => {
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'idle':
        return {
          label: 'Ready to Monitor',
          color: 'border-slate-200 bg-slate-50 text-slate-600',
          indicator: 'bg-slate-400'
        }
      case 'recording':
        return {
          label: 'Recording Audio',
          color: 'border-blue-200 bg-blue-50 text-blue-800',
          indicator: 'bg-blue-500 animate-pulse'
        }
      case 'cry-detected':
        return {
          label: 'Cry Detected',
          color: 'border-amber-200 bg-amber-50 text-amber-800',
          indicator: 'bg-amber-500 animate-bounce'
        }
      case 'analyzing':
        return {
          label: 'Analyzing Cry Pattern',
          color: 'border-violet-200 bg-violet-50 text-violet-800',
          indicator: 'bg-violet-500 animate-pulse'
        }
      case 'results':
        return {
          label: 'Analysis Complete',
          color: 'border-emerald-200 bg-emerald-50 text-emerald-800',
          indicator: 'bg-emerald-500'
        }
      default:
        return {
          label: 'Unknown Status',
          color: 'border-slate-200 bg-slate-50 text-slate-600',
          indicator: 'bg-slate-400'
        }
    }
  }

  const config = getStatusConfig(status)

  return (
    <div
      ref={ref}
      className={cn(
        'flex items-center gap-3 px-4 py-3 rounded-lg border-2 transition-all duration-300',
        config.color,
        className
      )}
      {...props}
    >
      <div className={cn('w-3 h-3 rounded-full', config.indicator)} />
      <span className="font-medium text-sm">{config.label}</span>
    </div>
  )
})
StatusIndicator.displayName = 'StatusIndicator'

export { StatusIndicator }