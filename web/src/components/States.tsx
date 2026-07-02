export function Loading({ label = "Loading…" }: { label?: string }) {
  return <div className="state state--loading">{label}</div>;
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="state state--error" role="alert">
      {message}
    </div>
  );
}

export function InfoState({ message }: { message: string }) {
  return <div className="state state--info">{message}</div>;
}
