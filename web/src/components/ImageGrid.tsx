import type { ImageItem } from "../types";
import { InfoState } from "./States";

interface Props {
  items: ImageItem[];
  urlFor: (file: string) => string;
  emptyMessage: string;
}

export function ImageGrid({ items, urlFor, emptyMessage }: Props) {
  if (items.length === 0) return <InfoState message={emptyMessage} />;
  return (
    <div className="img-grid">
      {items.map((item) => (
        <figure className="img-card" key={item.file}>
          <img src={urlFor(item.file)} alt={item.name} loading="lazy" />
          <figcaption className="cap">{item.name}</figcaption>
        </figure>
      ))}
    </div>
  );
}
