import React from "react";

/**
 * Renders markdown-like text with support for bold, lists, 
 * and LaTeX math conversion.
 */
export default function MarkdownText({ text }) {
  const convertLatexToText = (str) => {
    return str
      .replace(/\\\[([\s\S]+?)\\\]/g, (_, math) => convertLatexMath(math))
      .replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => convertLatexMath(math))
      .replace(/\\\((.+?)\\\)/g, (_, math) => convertLatexMath(math))
      .replace(/\$([^$\n]+?)\$/g, (_, math) => convertLatexMath(math));
  };

  const convertLatexMath = (math) => {
    return math
      .replace(/\n/g, " ")
      .replace(/\s+/g, " ")
      .replace(/\\times/g, "×")
      .replace(/\\div/g, "÷")
      .replace(/\\pm/g, "±")
      .replace(/\\mp/g, "∓")
      .replace(/\\cdot/g, "·")
      .replace(/\\ast/g, "*")
      .replace(/\\star/g, "★")
      .replace(/\\leq/g, "≤")
      .replace(/\\geq/g, "≥")
      .replace(/\\neq/g, "≠")
      .replace(/\\approx/g, "≈")
      .replace(/\\equiv/g, "≡")
      .replace(/\\lt/g, "<")
      .replace(/\\gt/g, ">")
      .replace(/\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}/g, "($1 / $2)")
      .replace(/\\sqrt\s*\{([^}]+)\}/g, "√($1)")
      .replace(/\\sqrt\s*(\d+)/g, "√$1")
      .replace(/\^{2}/g, "²")
      .replace(/\^2/g, "²")
      .replace(/\^{3}/g, "³")
      .replace(/\^3/g, "³")
      .replace(/\^\{([^}]+)\}/g, "^($1)")
      .replace(/_\{([^}]+)\}/g, "_$1")
      .replace(/\\alpha/g, "α")
      .replace(/\\beta/g, "β")
      .replace(/\\gamma/g, "γ")
      .replace(/\\delta/g, "δ")
      .replace(/\\epsilon/g, "ε")
      .replace(/\\pi/g, "π")
      .replace(/\\theta/g, "θ")
      .replace(/\\lambda/g, "λ")
      .replace(/\\mu/g, "μ")
      .replace(/\\sigma/g, "σ")
      .replace(/\\omega/g, "ω")
      .replace(/\\sum/g, "Σ")
      .replace(/\\prod/g, "Π")
      .replace(/\\rightarrow/g, "→")
      .replace(/\\leftarrow/g, "←")
      .replace(/\\Rightarrow/g, "⇒")
      .replace(/\\Leftarrow/g, "⇐")
      .replace(/\\infty/g, "∞")
      .replace(/\\partial/g, "∂")
      .replace(/\\nabla/g, "∇")
      .replace(/\\therefore/g, "∴")
      .replace(/\\because/g, "∵")
      .replace(/\\text\s*\{([^}]+)\}/g, "$1")
      .replace(/\\mathrm\s*\{([^}]+)\}/g, "$1")
      .replace(/\\mathbf\s*\{([^}]+)\}/g, "$1")
      .replace(/\\[a-zA-Z]+/g, "")
      .replace(/\{/g, "")
      .replace(/\}/g, "")
      .replace(/\s+/g, " ")
      .trim();
  };

  const parseBold = (str) => {
    const parts = [];
    let key = 0;
    const regex = /\*\*(.+?)\*\*|__(.+?)__/g;
    let lastIndex = 0;
    let match;
    while ((match = regex.exec(str)) !== null) {
      if (match.index > lastIndex) {
        parts.push(str.slice(lastIndex, match.index));
      }
      parts.push(
        <strong key={key++} className="font-semibold">
          {match[1] || match[2]}
        </strong>
      );
      lastIndex = regex.lastIndex;
    }
    if (lastIndex < str.length) {
      parts.push(str.slice(lastIndex));
    }
    return parts.length > 0 ? parts : str;
  };

  const processedText = convertLatexToText(text);
  const lines = processedText.split("\n");
  const elements = [];
  let currentList = [];
  let listType = null;
  let key = 0;

  const flushList = () => {
    if (currentList.length > 0) {
      if (listType === "ol") {
        elements.push(
          <ol key={key++} className="list-decimal list-inside space-y-1 my-2 ml-2">
            {currentList.map((item, i) => (
              <li key={i} className="leading-relaxed">{parseBold(item)}</li>
            ))}
          </ol>
        );
      } else {
        elements.push(
          <ul key={key++} className="list-disc list-inside space-y-1 my-2 ml-2">
            {currentList.map((item, i) => (
              <li key={i} className="leading-relaxed">{parseBold(item)}</li>
            ))}
          </ul>
        );
      }
      currentList = [];
      listType = null;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();

    const numberedMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
    if (numberedMatch) {
      if (listType !== "ol") { flushList(); listType = "ol"; }
      currentList.push(numberedMatch[2]);
      continue;
    }

    const bulletMatch = trimmed.match(/^[*\-•]\s+(.*)/);
    if (bulletMatch) {
      if (listType !== "ul") { flushList(); listType = "ul"; }
      currentList.push(bulletMatch[1]);
      continue;
    }

    flushList();

    if (!trimmed) {
      elements.push(<div key={key++} className="h-2" />);
      continue;
    }

    elements.push(
      <p key={key++} className="leading-relaxed">
        {parseBold(trimmed)}
      </p>
    );
  }

  flushList();

  return <div className="space-y-1">{elements}</div>;
}
