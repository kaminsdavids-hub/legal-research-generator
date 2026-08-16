"use client";

import type { ReactNode } from "react";

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

export function Card({
  title,
  icon,
  actions,
  children,
  className,
}: {
  title?: string;
  icon?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cx("flex flex-col rounded-xl border border-slate-200 bg-white shadow-sm", className)}>
      {(title || actions) && (
        <header className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
            {icon}
            {title}
          </div>
          <div className="flex items-center gap-2">{actions}</div>
        </header>
      )}
      <div className="min-h-0 flex-1 overflow-auto scroll-thin p-4">{children}</div>
    </section>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled,
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "subtle" | "danger";
  disabled?: boolean;
  title?: string;
}) {
  const styles: Record<string, string> = {
    primary: "bg-accent text-white hover:bg-violet-700",
    ghost: "border border-slate-300 text-slate-700 hover:bg-slate-50",
    subtle: "bg-slate-100 text-slate-700 hover:bg-slate-200",
    danger: "bg-rose-50 text-rose-700 hover:bg-rose-100 border border-rose-200",
  };
  return (
    <button
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={cx(
        "inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50",
        styles[variant]
      )}
    >
      {children}
    </button>
  );
}

const STATUS_COLORS: Record<string, string> = {
  verified: "bg-emerald-100 text-emerald-700",
  cited: "bg-sky-100 text-sky-700",
  drafted: "bg-amber-100 text-amber-700",
  idea: "bg-slate-100 text-slate-600",
  pending: "bg-amber-100 text-amber-700",
  needs_review: "bg-orange-100 text-orange-700",
  removed: "bg-rose-100 text-rose-700",
  keep: "bg-emerald-100 text-emerald-700",
  proposed: "bg-slate-100 text-slate-600",
  cut: "bg-rose-100 text-rose-700",
  supporting: "bg-emerald-100 text-emerald-700",
  contrary: "bg-rose-100 text-rose-700",
  neutral: "bg-slate-100 text-slate-600",
};

export function Badge({ value }: { value: string }) {
  return (
    <span
      className={cx(
        "rounded-full px-2 py-0.5 text-xs font-medium capitalize",
        STATUS_COLORS[value] ?? "bg-slate-100 text-slate-600"
      )}
    >
      {value.replace("_", " ")}
    </span>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="grid h-full place-items-center text-center text-sm text-slate-400">{children}</div>;
}
