import * as React from "react"

import { cn } from "@/lib/utils"

const Progress = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & {
    value?: number
    max?: number
    indicatorClassName?: string
  }
>(({ className, value = 0, max = 100, indicatorClassName, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "relative h-4 w-full overflow-hidden rounded-full bg-secondary",
      className
    )}
    {...props}
  >
    <div
      className={cn(
        "h-full w-full flex-1 bg-primary transition-all",
        indicatorClassName
      )}
      style={{ transform: `translateX(-${100 - (value / max) * 100}%)` }}
    />
  </div>
))
Progress.displayName = "Progress"

const WaveformBar = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & {
    value?: number
    max?: number
    height?: string
    isActive?: boolean
  }
>(({ className, value = 0, max = 100, height = "4px", isActive = false, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "w-full rounded-t-sm transition-all duration-100 bg-muted",
      isActive && "bg-primary/60",
      className
    )}
    style={{
      height: `${Math.max((value / max) * 100, parseInt(height))}%`,
      minHeight: height,
    }}
    {...props}
  />
))
WaveformBar.displayName = "WaveformBar"

export { Progress, WaveformBar }