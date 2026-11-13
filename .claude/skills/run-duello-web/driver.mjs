// Headless Chrome driven over HTTP, so each step is one curl and the browser
// state survives between them. egui draws on one canvas: act by coordinates,
// read screenshots.
//   GET localhost:9333/cmd?op=nav&url=...      load a page
//   op=click&x=&y=[&n=2]  op=move&x=&y=  op=drag&x=&y=&x2=&y2=
//   op=type&text=  op=key&key=Meta+a  op=wait&ms=
//   op=upload&path=  (arms the next file chooser; then trigger it)
//   op=shot&name=    (prints the PNG path)  op=logs  op=quit
// Output goes to $DRIVER_OUT (default ./run-out): shots/, downloads/.
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const out = path.resolve(process.env.DRIVER_OUT ?? 'run-out');
for (const d of ['shots', 'downloads']) fs.mkdirSync(path.join(out, d), { recursive: true });

// The installed Chrome: no Playwright browser download, and it has WebGPU.
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
const page = await ctx.newPage();

const logs = [];
const noise = /integrity|willReadFrequently/;
page.on('console', m => {
  const t = `${m.type()}: ${m.text()}`;
  if (['error', 'warning'].includes(m.type()) && !noise.test(t)) logs.push(t.slice(0, 300));
});
page.on('pageerror', e => logs.push('pageerror: ' + String(e).slice(0, 300)));

let upload = null;
page.on('filechooser', fc => { if (upload) { fc.setFiles(upload); upload = null; } });
page.on('download', async d => {
  const p = path.join(out, 'downloads', d.suggestedFilename());
  await d.saveAs(p);
  logs.push('download: ' + p);
});

const ops = {
  nav: async q => { await page.goto(q.url); },
  click: q => page.mouse.click(+q.x, +q.y, { clickCount: +(q.n || 1) }),
  move: q => page.mouse.move(+q.x, +q.y),
  drag: async q => {
    await page.mouse.move(+q.x, +q.y);
    await page.mouse.down();
    await page.mouse.move(+q.x2, +q.y2, { steps: 10 });
    await page.mouse.up();
  },
  type: q => page.keyboard.type(q.text, { delay: 20 }),
  // A query string turns '+' into a space, so Meta+a arrives as 'Meta a'.
  key: q => page.keyboard.press(q.key.replace(/ /g, '+')),
  upload: q => { upload = q.path; },
  wait: q => page.waitForTimeout(+q.ms),
  shot: async q => {
    const p = path.join(out, 'shots', q.name + '.png');
    await page.screenshot({ path: p });
    return p;
  },
  logs: () => logs.splice(0).join('\n') || '(none)',
};

http.createServer(async (req, res) => {
  const q = Object.fromEntries(new URL(req.url, 'http://x').searchParams);
  if (q.op === 'quit') {
    res.end('bye');
    await browser.close();
    process.exit(0);
  }
  try {
    const op = ops[q.op];
    res.end(op ? String((await op(q)) ?? 'ok') : 'ERR unknown op');
  } catch (e) {
    res.end('ERR ' + e);
  }
}).listen(9333, () => console.log('ready on :9333'));
