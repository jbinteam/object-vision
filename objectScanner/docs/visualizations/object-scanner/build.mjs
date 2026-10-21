import {readFileSync,writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {motionInstrument} from './motion-instrument.mjs';
import {motionView} from './motion-view.mjs';
import {datasetMotion} from './motion-dataset.mjs';
import {poseMotion} from './motion-pose.mjs';
import {streams,artifacts,contracts} from './model.mjs';
import {datasetDeepDive} from './deep-dataset.mjs';
import {poseDeepDive} from './deep-pose.mjs';
import {deepView} from './deep-view.mjs';
const lessons=[...datasetDeepDive,...poseDeepDive];
import * as k from './vendor/iso-kit.mjs';
const here=fileURLToPath(new URL('.',import.meta.url));
const theme=process.argv.includes('--light')?'light':'dark';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const figures=streams.map((s,i)=>motionInstrument(s.id,(i===0?datasetMotion:poseMotion)(0,0).body,process.argv.includes('--audit')));
const deepFigure=motionInstrument('deep',datasetMotion(0,0).body);
const deepMarkup=motionView('deep-motion',deepFigure,0,streams[0].stages[0]);
const stageButtons=(s,i)=>s.stages.map((v,j)=>`<button class="step ${j===0?'selected':''}" data-stream="${i}" data-step="${j}" aria-pressed="${j===0}"><span class="step-number">0${j+1}</span><span>${esc(v.short)}</span></button>`).join('');
const body=`<header><div class="masthead"><span>OBJECTSCANNER / SYSTEM ANATOMY</span><div class="tools"><button id="tour" aria-pressed="true">Pause tour</button><button id="theme" aria-label="Switch color theme">${theme==='dark'?'Light':'Dark'} mode</button></div></div><h1>From captured views<br>to an object pose.</h1><p class="intro">Two streams. One calibrated object frame.<br>Prepare the data once, then estimate pose from each new RGB-D observation.</p></header>
<div class="streams">${streams.map((s,i)=>`<section class="stream" id="${s.id}" aria-labelledby="title-${i}"><div class="stream-heading"><span class="stream-number">0${i+1}</span><div><span class="eyebrow">${i===0?'OFFLINE · PREPARE':'RUNTIME · ESTIMATE'}</span><h2 id="title-${i}">${esc(s.title)}</h2></div></div><p class="subtitle">${esc(s.subtitle)}</p>${motionView('motion-'+i,figures[i],i,s.stages[0])}<div class="stage-controls" aria-label="${esc(s.title)} stages">${stageButtons(s,i)}</div><article class="detail" aria-labelledby="stage-title-${i}"><div class="detail-top"><span class="eyebrow">SELECTED STAGE</span><span class="detail-count">01 / 06</span></div><h3 id="stage-title-${i}">${esc(s.stages[0].title)}</h3><p class="stage-description">${esc(s.stages[0].detail)}</p><dl><dt>IN</dt><dd class="stage-input">${esc(s.stages[0].input)}</dd><dt>OUT</dt><dd class="stage-output">${esc(s.stages[0].output)}</dd></dl><button class="deep-open" aria-controls="deep-dive" data-deep-open="${i}">Explore this step in depth <span aria-hidden="true">↘</span></button></article><p class="branch">${i===0?'<strong>Two outputs, in parallel.</strong> Reviewed captures feed paired training datasets and a calibrated TSDF reference. The reference is measured geometry.':'<strong>A guarded tracking loop.</strong> Startup / lost → PnP seed. Tracking → last accepted raw pose. GICP rejection preserves the seed; repeated failure returns to recovery.'}</p></section>`).join('')}</div>
${deepView(streams,lessons,deepMarkup)}
<section class="handoff"><div class="section-top"><span class="eyebrow">THE HANDOFF BETWEEN STREAMS</span><span class="bundle-count">4 linked artifacts</span></div><h2>A bundle that agrees on geometry.</h2><div class="artifacts">${artifacts.map((a,i)=>`<article><span class="artifact-index">${String(i+1).padStart(2,'0')} /</span><h3>${esc(a.name)}</h3><p>${esc(a.detail)}</p></article>`).join('')}</div></section>
<details class="contracts"><summary>Coordinate contract & implementation sources <span>+</span></summary><div class="contract-body"><ul>${contracts.map(c=>`<li>${esc(c)}</li>`).join('')}</ul><p>Read the <a href="../../../scripts/collect_process/README.md">collection workflow</a>, <a href="../../PHASE4_YOLO_PNP_GICP.md">Phase 4 guide</a> and <a href="../../../ros2_ws/src/object_pose_tracking/config/phase4.yaml">runtime configuration</a>.</p></div></details><footer><span>Illustrative sequence · not live telemetry</span><span>RGB-D · YOLO · PnP · GICP</span></footer><div class="sr-only" id="announcement" aria-live="polite"></div>`;
const css=readFileSync(here+'style.css','utf8');
const runtime=readFileSync(here+'live.js','utf8');
const model=readFileSync(here+'model.mjs','utf8').replace(/^export /gm,'');
const inlineMotion=(file,name)=>`const ${name}=(()=>{${readFileSync(here+file,'utf8').replace(/^export /gm,'')};return ${name};})();`;
const script=model+'\nconst offsets='+JSON.stringify(figures.map(f=>f.offset))+';\n'+inlineMotion('motion-dataset.mjs','datasetMotion')+'\n'+inlineMotion('motion-pose.mjs','poseMotion')+'\n'+readFileSync(here+'motion-live.js','utf8')+'\n'+runtime;
const html=k.pageHtml({title:'objectScanner — two streams',theme,width:1280,body,script}).replace('</style>',css+'</style>');
const file=theme==='light'?'index-light.html':'index.html';
writeFileSync(here+file,html);
console.log('Wrote '+file+' · '+figures.map(f=>f.count).join(' + ')+' solids');
