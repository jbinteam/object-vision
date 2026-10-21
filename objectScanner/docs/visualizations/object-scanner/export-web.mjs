// Portable publishing bundle: the figure plus only explicitly linked evidence.
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve,dirname,relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {datasetDeepDive} from './deep-dataset.mjs';
import {poseDeepDive} from './deep-pose.mjs';
const here=fileURLToPath(new URL('.',import.meta.url)),root=resolve(here,'../../..');
const dest=resolve(process.argv[2]||'output/object-pose-web');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sources=new Map();
for(const lesson of [...datasetDeepDive,...poseDeepDive])for(const source of lesson.sources){
 if(!sources.has(source.path))sources.set(source.path,new Set());
 const selected=sources.get(source.path);
 for(const range of source.lines.split(',')){const [start,end]=range.match(/\d+/g).map(Number);for(let line=start;line<=(end??start);line++)selected.add(line);}
}
const guides=['scripts/collect_process/README.md','docs/PHASE4_YOLO_PNP_GICP.md'];
for(const path of guides)sources.set(path,new Set(readFileSync(resolve(root,path),'utf8').split('\n').map((_,i)=>i+1)));
const shell=(title,body)=>`<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${esc(title)} · Object pose sources</title><style>body{background:#f8f8f6;color:#252a30;font:15px/1.7 system-ui,sans-serif;margin:30px auto;padding:0 20px;max-width:1100px}a{color:#205f81}h1{font-size:24px;overflow-wrap:anywhere}pre{overflow:auto;background:white;border:1px solid #d8dce0;padding:16px;font:12px/1.8 monospace}code{display:block;min-width:max-content}code:target{background:#fff1b8}.line{display:inline-block;width:5ch;color:#68737a;user-select:none}.gap{color:#68737a}p{max-width:80ch}</style>${body}</html>`;
const manifest={generatedAt:new Date().toISOString(),sources:[]};
for(const [path,selected] of sources){
 const lines=readFileSync(resolve(root,path),'utf8').split('\n'),ordered=[...selected].sort((a,b)=>a-b);let previous=0,code='';
 for(const line of ordered){if(line>previous+1)code+='<span class="gap">… uncited lines omitted …</span>\n';code+=`<code id="L${line}"><a class="line" href="#L${line}">${line}</a>${esc(lines[line-1])}</code>`;previous=line;}
 const output=resolve(dest,'sources',path+'.html');mkdirSync(dirname(output),{recursive:true});
 const back='../'.repeat(path.split('/').length)+'index-light.html';
 writeFileSync(output,shell(path,`<a href="${back}">← Back to the animated process</a><h1>${esc(path)}</h1><p>Implementation snapshot supporting the published lesson. ${guides.includes(path)?'Workflow guide.':'Only cited line ranges are included; this is not a runnable source file.'} Configuration and recorded examples describe this project snapshot.</p><pre>${code}</pre>`));
 manifest.sources.push({path,lines:ordered.length});
}
for(const file of ['index.html','index-light.html']){
 let html=readFileSync(here+file,'utf8');
 html=html.replace(/href="((?:\.\.\/)+[^"]+)"/g,(_,href)=>{const path=relative(root,resolve(here,href));if(!sources.has(path))throw Error('Unpackaged source '+path);return `href="sources/${path}.html"`;});
 if(/href="\.\.\//.test(html))throw Error('Nonportable relative source link remains');
 mkdirSync(dest,{recursive:true});writeFileSync(resolve(dest,file),html);
}
writeFileSync(resolve(dest,'source-manifest.json'),JSON.stringify(manifest,null,2));
console.log(`Exported both figures and ${sources.size} linked source snapshots to ${dest}`);
