#!/usr/bin/env node
// Render at build time. No JavaScript, CDN or math fonts are needed in the EPUB.
const fs = require('node:fs');
const path = require('node:path');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
require('mathjax-full/js/input/tex/ams/AmsConfiguration.js');

try {
  const [, , requestPath, outputDir] = process.argv;
  if (!requestPath || !outputDir) throw new Error('usage: render_math.cjs INPUT.json OUTPUT_DIR');
  const adaptor = liteAdaptor();
  RegisterHTMLHandler(adaptor);
  const tex = new TeX({packages: ['base', 'ams'], formatError: (_jax, error) => {throw error;}});
  const document = mathjax.document('', {InputJax: tex, OutputJax: new SVG({fontCache: 'none'})});
  fs.mkdirSync(outputDir, {recursive: true});
  const results = [];
  for (const item of JSON.parse(fs.readFileSync(requestPath, 'utf8'))) {
    if (!/^[a-f0-9]{24}$/.test(item.id)) throw new Error('invalid formula id');
    try {
      const container = document.convert(item.tex, {display: item.display});
      const svg = adaptor.firstChild(container);
      // liteAdaptor serializes HTML attributes, not arbitrary TeX as XML.
      // Base64 keeps alignment '&', comparisons '<', and quotes XML-safe.
      adaptor.setAttribute(svg, 'data-tex-b64', Buffer.from(item.tex, 'utf8').toString('base64'));
      adaptor.setAttribute(svg, 'data-display', item.display ? 'block' : 'inline');
      const markup = adaptor.outerHTML(svg);
      if (/data-mjx-error|data-mml-node="merror"|<text\b/.test(markup)) {
        throw new Error('formula contains a typesetting error or a font-dependent glyph');
      }
      fs.writeFileSync(path.join(outputDir, `${item.id}.svg`), markup);
      results.push({id: item.id, width: adaptor.getAttribute(svg, 'width'),
        height: adaptor.getAttribute(svg, 'height'),
        vertical_align: adaptor.getStyle(svg, 'vertical-align') || '0ex'});
    } catch (error) {
      throw new Error(`Formula ${item.id} (${item.tex}): ${error.message}`);
    }
  }
  process.stdout.write(JSON.stringify(results));
} catch (error) {
  process.stderr.write(`${error.message}\n`);
  process.exitCode = 1;
}
