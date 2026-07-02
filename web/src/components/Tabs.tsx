interface Props {
  tabs: string[];
  active: number;
  onChange: (index: number) => void;
}

export function Tabs({ tabs, active, onChange }: Props) {
  return (
    <div className="tabbar" role="tablist">
      {tabs.map((label, i) => (
        <button
          key={label}
          role="tab"
          aria-selected={active === i}
          className={`tab ${active === i ? "tab--active" : ""}`}
          onClick={() => onChange(i)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
