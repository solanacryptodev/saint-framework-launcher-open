import { JSX } from "solid-js";

interface InfoContainerProps {
  title: string;
  children: JSX.Element;
  class?: string;
}

export default function InfoContainer(props: InfoContainerProps) {
  return (
    <div class={`info-container ${props.class || ""}`}>
      <div class="info-container-header">
        <h3 class="info-container-title">{props.title}</h3>
      </div>
      <div class="info-container-content">
        {props.children}
      </div>
    </div>
  );
}
