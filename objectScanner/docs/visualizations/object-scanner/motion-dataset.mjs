// Deterministic educational displays, not live capture or training telemetry.
// Contracts/defaults: turntable_dataset.py; scripts/capture_turntable_dataset.py;
// scripts/augment_centerpose_backgrounds.py; generate_yolo_seg_pose_dataset.py;
// build_calibrated_reference.py; pnp_snapshot.py; train_yolo_seg_pose.py.
const clamp = value => Math.max(0, Math.min(1, Number.isFinite(Number(value)) ? Number(value) : 0));
const f = value => Number(value.toFixed(2));
const ease = value => { const t = clamp(value); return t * t * (3 - 2 * t); };
const between = (p, start, end) => ease((p - start) / (end - start));
const txt = (x, y, text, small = false, attrs = '') => `<text x="${f(x)}" y="${f(y)}" class="${small ? 'm-small' : 'm-label'}" stroke="none" ${attrs}>${text}</text>`;
const line = (x1, y1, x2, y2, cls = 'm-ink', attrs = '') => `<path d="M${f(x1)} ${f(y1)}L${f(x2)} ${f(y2)}" class="${cls}" fill="none" ${attrs}/>`;
const rect = (x, y, w, h, cls = 'm-outline', attrs = '') => `<rect x="${f(x)}" y="${f(y)}" width="${f(w)}" height="${f(h)}" rx="4" class="${cls}" ${attrs}/>`;
const dot = (x, y, radius = 2.5, cls = 'm-lit', attrs = '') => `<circle cx="${f(x)}" cy="${f(y)}" r="${radius}" class="${cls}" ${attrs}/>`;
const path = (d, cls = 'm-ink', attrs = '') => `<path d="${d}" class="${cls}" ${attrs}/>`;
const poly = (points, cls = 'm-ink', attrs = '') => `<polygon points="${points.map(([x,y])=>`${f(x)},${f(y)}`).join(' ')}" class="${cls}" ${attrs}/>`;
const iso = (x,y,z, cx=230,cy=155) => [cx + (x-y)*0.8, cy+(x+y)*0.35-z];
const corners = (cx=230,cy=155,w=100,d=80,h=60) => Array.from({length:8},(_,i)=>iso((i&1?1:-1)*w/2,(i&2?1:-1)*d/2,(i&4?1:-1)*h/2,cx,cy));
const edges = [[0,1],[0,2],[1,3],[2,3],[0,4],[1,5],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
function cube(points, count=12, cls='m-ink') {
  return edges.slice(0,count).map(([a,b])=>line(...points[a],...points[b],cls)).join('');
}
function tag(x,y,id,opacity=1) {
  return `<g opacity="${f(opacity)}">${rect(x-11,y-11,22,22,'m-surface')}${rect(x-9,y-9,18,18,'m-fill')}${rect(x-6,y-6,12,12,'m-surface')}${rect(x-4,y-4,5,5,'m-fill')}${rect(x+1,y+1,4,4,'m-fill')}${txt(x+15,y+4,String(id),true)}</g>`;
}
function progress(p, labels) {
  const n = labels.length;
  return labels.map((label,i)=>{
    const x = 20+i*460/n;
    return `${rect(x,225,460/n-8,3,i<=Math.min(n-1,Math.floor(p*n))?'m-lit':'m-fill')}${txt(x,243,label,true)}`;
  }).join('');
}
function calibration(p) {
  const pts = corners(246,115,110,72,65);
  const resolve = between(p,.12,.36), frame = between(p,.33,.64), project = between(p,.6,.94);
  const origin = pts[0];
  let body = txt(18,22,'CALIBRATED BOX · ILLUSTRATIVE GEOMETRY');
  body += poly([iso(-95,-60,-33,246,115),iso(95,-60,-33,246,115),iso(95,60,-33,246,115),iso(-95,60,-33,246,115)],'m-surface');
  body += poly([pts[0],pts[1],pts[3],pts[2]],'m-fill','opacity="0.5"');
  body += tag(...pts[0],1,1-project*.8)+tag(...pts[1],2,1-project*.8)+tag(...pts[2],3,1-project*.8);
  body += tag(246,82.5,4,1-project*.8);
  body += tag(99,154,5)+tag(370,154,6);
  body += `<g opacity="${f(resolve)}">${line(...origin,...pts[1],'m-soft')}${line(...origin,...pts[2],'m-soft')}${txt(310,198,'X / Y resolved',true)}</g>`;
  body += `<g opacity="${f(frame)}">${cube(pts,12,'m-ink')}${dot(246,115,3)}${line(246,115,294,136)}${line(246,115,208,132)}${line(246,115,246,71)}${txt(296,141,'X',true)}${txt(194,141,'Y',true)}${txt(251,66,'Z',true)}</g>`;
  body += pts.map(([x,y],i)=>`<g opacity="${f(project)}">${dot(x,y,2.7)}${txt(x+(i&1?6:-13),y+(i&4?-8:14),String(i),true)}</g>`).join('');
  body += txt(18,52,'14 mm setup tags',true)+txt(18,69,'28 mm table tags',true);
  body += txt(328,49,'box_center',true)+txt(328,66,'XYZ sign order',true);
  body += txt(18,208,'T_camera_object = T_camera_table × T_table_object',true);
  body += progress(p,['Detect tags','Resolve roles','Set frame','8 corners']);
  return {body,phase:p<.25?'Identify setup and permanent tags':p<.5?'Resolve base axes and top height':p<.75?'Place the box-center frame':'Preserve all eight corner identities',readout:p<.25?'4 setup tags + 2 mapped table tags':p<.5?'Orthogonal X / Y; Z toward top':p<.75?'Origin at the calibrated box center':'0 −−−  ·  7 +++',caption:'Demonstration: setup tags establish the box; fixed table tags recover the same frame from later views.'};
}
function capture(p) {
  const sample = 'view_007';
  const tx = .02;
  const recorded = p>=.8;
  let body = txt(18,22,'ONE VIEW · THREE MATCHED RECORDS');
  const cells = [[18,'RGB'],[174,'ALIGNED DEPTH'],[330,'POSE + K']];
  for(const [x,name] of cells)body+=rect(x,45,132,140,'m-surface')+txt(x+8,64,name,true);
  body+=rect(28,74,112,99,'m-fill','opacity="0.45"');
  // Exact pinhole projection for the numeric R/t example shown in the third pane.
  const pts=Array.from({length:8},(_,i)=>{
    const x=(i&1?.05:-.05)+.02, y=(i&2?.04:-.04)-.01, z=(i&4?.03:-.03)+.5;
    return [28+(600*x/z+320)*.175,82+(600*y/z+240)*.175];
  });
  body+=poly([pts[0],pts[1],pts[5],pts[4]],'m-soft','fill="none"')+poly([pts[4],pts[5],pts[7],pts[6]],'m-fill')+cube(pts);
  if(p>=.25)body+=pts.map(point=>dot(...point,1.6)).join('');
  // Depth bands are symbolic gray levels, never an inferred error/accuracy map.
  for(let i=0;i<7;i++)body+=rect(185+i*14,76,14,97,'m-fill',`opacity="${f(.14+i*.09)}"`);
  const depthPts=pts.map(([x,y])=>[x+156,y]);
  body+=poly([0,1,5,7,6,2].map(i=>depthPts[i]),'m-outline','fill="none"');
  body+=dot(244.2,121.9,3)+txt(193,168,'500 → 0.500 m',true);
  body+=txt(340,86,'T_camera_object',true)+txt(340,106,'R = identity',true)+txt(340,123,`t = [${tx.toFixed(2)},`,true)+txt(340,139,'     −0.01, 0.50] m',true)+txt(340,160,'K + 8 projections',true);
  const gate = between(p,.48,.76);
  body+=rect(18,193,444,21,'m-outline','fill="none"')+rect(19,194,442*gate,19,'m-fill','opacity="0.2"');
  body+=txt(26,207,recorded?`${sample} · RGB + depth + label saved`:'Save gate: valid tag pose · setup tags removed',true);
  body+=progress(p,['Read pair','Project box','Check gates','Save view']);
  return {body,phase:p<.25?'Read one RGB-D observation':p<.5?'Derive pose and projected corners':p<.8?'Check pose and setup-tag gates':'Save matched evidence and labels',readout:recorded?`${sample}: image / depth / JSON`:'Illustrative depth unit: 1 count = 1 mm',caption:'Illustrative saved view. Raw RGB is retained for audit; the training image has permanent tags inpainted.'};
}
const silhouette = 'M128 164L128 105Q130 88 147 88L208 88Q220 90 220 105L220 164Q217 177 204 177L144 177Q130 175 128 164Z';
function masks(p) {
  const step=Math.min(3,Math.floor(p*4));
  const τ=.5;
  let body=txt(18,22,'FOREGROUND · FROM PROBABILITY TO REVIEW');
  body+=rect(18,44,276,169,'m-surface')+rect(100,69,147,123,'m-outline','fill="none" stroke-dasharray="4 3"');
  body+=txt(28,61,step===0?'BiRefNet probability':step===1?'threshold + pose crop':step===2?'exclude + disconnect':'selected component',true);
  // Deterministic grid. Its values demonstrate probabilities, not model metrics.
  if(step===0) {
    for(let row=0;row<11;row++)for(let col=0;col<14;col++) {
      const x=40+col*17,y=76+row*11;
      const inObject=x>=128&&x<=220&&y>=88&&y<=177;
      const branch=x>=213&&x<=264&&y>=123&&y<=139;
      const prob=inObject?.82:branch?.63:.12+((row*7+col*3)%4)*.05;
      body+=rect(x,y,14,8,'m-fill',`opacity="${f(prob)}"`);
    }
  } else {
    body+=path(silhouette,'m-lit');
    const prune=between(p,.4,.76);
    body+=`<g opacity="${f(1-prune)}">${rect(215,124,55,10,'m-fill')}${rect(104,91,31,24,'m-fill')}${rect(239,163,20,20,'m-fill')}</g>`;
    if(step>=2)body+=rect(104,91,31,24,'m-outline','fill="none"')+line(107,94,132,112,'m-dim')+line(132,94,107,112,'m-dim');
    if(step===3)body+=path(silhouette,'m-outline','fill="none"');
  }
  body+=dot(175,132,3,'m-surface')+line(170,132,180,132)+line(175,127,175,137);
  body+=txt(310,65,`τ = ${τ.toFixed(1)}`,false)+txt(310,88,'Pose-guided crop',true)+txt(310,107,'Subtract tag pixels',true)+txt(310,126,'11 px opening',true)+txt(310,145,'Center component',true)+txt(310,164,'Inspect silhouette',true);
  body+=txt(310,198,step===3?'REVIEW PREVIEW':'ILLUSTRATIVE FIELD',true);
  body+=progress(p,['Probability','Threshold','Refine','Review']);
  return {body,phase:['Estimate foreground probabilities','Threshold inside the pose crop','Remove exclusions and thin connections','Review the selected object mask'][step],readout:step===0?'Example field; not a measured prediction':step===1?'Threshold τ = 0.5':step===2?'Tag exclusion + morphological opening':'Center component retained; preview required',caption:'Threshold stays at 0.5 in this example. The real script lowers it only when refinement fails, then can retry full-frame inference.'};
}
function exportLabels(p) {
  const step=Math.min(3,Math.floor(p*4));
  const pts=corners(130,135,85,56,50);
  let body=txt(18,22,'PAIRED LABELS · ONE SOURCE ID');
  body+=rect(18,44,224,164,'m-surface')+rect(258,44,204,164,'m-surface');
  body+=txt(28,63,'source_007 · 640 × 480',true);
  body+=path('M83 162L83 109Q85 88 102 88L156 88Q178 91 178 111L178 159Q175 177 154 177L104 177Q85 176 83 162Z','m-fill');
  if(step>=1)body+=cube(pts,Math.min(12,Math.floor(between(p,.22,.51)*12)),'m-soft');
  if(step>=1)body+=pts.map(([x,y],i)=>`${dot(x,y,2.3)}${txt(x+5,y-5,String(i),true)}`).join('');
  body+=txt(269,65,step<2?'SAME IMAGE / SAME SPLIT':'SOURCE-GROUPED SPLIT',true);
  body+=txt(270,89,'seg: one silhouette polygon',true)+txt(270,111,'pose: eight ordered corners',true);
  body+=line(270,123,449,123,'m-dim');
  body+=txt(270,142,'(320, 240) px',true)+txt(270,160,'→ (0.5, 0.5, 2)',true)+txt(270,181,'off-image → (0, 0, 0)',true);
  if(step>=2){
    body+=rect(26,187,69,15,'m-fill','opacity="0.25"')+txt(32,198,'TRAIN',true);
    body+=rect(102,187,130,15,'m-outline','fill="none"')+txt(110,198,'source_007 held together',true);
  }
  body+=progress(p,['Mask contour','Corner IDs','Group split','Export pair']);
  return {body,phase:['Use the reviewed silhouette','Add eight canonical projections','Assign one split per source view','Write paired segmentation and pose labels'][step],readout:step<2?'source_007 → segmentation + pose':'Illustrative source_007: train in both tasks',caption:'In-image visibility 2 marks annotation availability, including hidden virtual corners. Polygon fidelity and canonical projections are checked on export.'};
}
function fusion(p) {
  const viewCount=Math.min(4,Math.floor(p*4)+1);
  const completed=between(p,.12,.94);
  let body=txt(18,22,'TSDF FUSION · COMMON OBJECT FRAME');
  const cx=242,cy=120;
  const pts=corners(cx,cy,103,73,62);
  body+=poly([iso(-90,-65,-34,cx,cy),iso(90,-65,-34,cx,cy),iso(90,65,-34,cx,cy),iso(-90,65,-34,cx,cy)],'m-surface');
  const cameras=[[70,80],[393,76],[407,177],[72,182]];
  cameras.forEach(([x,y],i)=>{
    body+=rect(x-14,y-9,28,18,i<viewCount?'m-fill':'m-outline',i<viewCount?'':'fill="none"')+dot(x+16,y,4,'m-outline','fill="none"');
    if(i<viewCount)body+=line(x+(x<cx?22:-22),y,cx+(x<cx?-56:56),cy+(y<cy?-28:28),'m-dim','stroke-dasharray="2 5"');
    body+=txt(x-17,y+25,`view ${i+1}`,true);
  });
  // A deterministic schematic surface accumulated by view; not TSDF measurements.
  const surface=[];
  for(let face=0;face<3;face++)for(let row=0;row<8;row++)for(let col=0;col<9;col++){
    const a=(col/8-.5),b=(row/7-.5);
    const world=face===0?[a*96,b*66,31]:face===1?[a*96,-36,b*60]:[51,a*66,b*60];
    const [x,y]=iso(...world,cx,cy);
    surface.push({x,y,rank:(face*31+row*7+col*11)%216});
  }
  const visible=Math.floor(surface.length*completed);
  surface.sort((a,b)=>a.rank-b.rank).slice(0,visible).forEach(({x,y})=>{body+=dot(x,y,1.35,'m-fill');});
  body+=cube(pts,12,'m-soft')+dot(cx,cy,2.5)+txt(cx-28,cy+53,'box_center',true);
  if(p>.7)body+=txt(151,201,'crop: half-dimensions + 3 mm',true);
  body+=txt(18,45,'4 example observations',true)+txt(319,45,'schematic surface',true);
  body+=progress(p,['Read views','Transform','Fuse depth','Crop cloud']);
  return {body,phase:p<.25?'Use saved poses and masked depth':p<.5?'Place observations in box_center':p<.75?'Accumulate the implicit surface':'Extract, crop and filter the reference',readout:`${viewCount} / 4 illustrative views admitted`,caption:'Four schematic observations explain fusion. The real builder defaults to at least eight views; voxel size is not a measured accuracy guarantee.'};
}
function training(p) {
  const draw=Math.min(12,Math.floor(p*12)+1);
  const step=Math.min(3,Math.floor(p*4));
  const background=Math.floor(p*6)%3;
  let body=txt(18,22,'TRAIN TWO MODELS · PIN ONE GEOMETRY');
  body+=rect(18,44,206,163,'m-surface')+txt(28,63,`source_007 · draw ${draw} / 12`,true);
  body+=rect(28,75,186,119,'m-fill','opacity="0.2"');
  for(let i=0;i<9;i++){
    const x=30+i*22;
    if(background===0)body+=line(x,78,x,190,'m-dim');
    if(background===1)body+=line(31,79+i*13,211,79+i*13,'m-dim');
    if(background===2)body+=rect(x,82+(i%3)*31,14,17,'m-fill','opacity="0.4"');
  }
  body+=path('M87 164L87 113Q88 99 104 99L143 99Q158 99 158 114L158 164Q157 178 144 178L101 178Q88 176 87 164Z','m-lit');
  body+=cube(corners(122,137,62,44,53),12,'m-outline');
  body+=txt(249,62,'PINNED GEOMETRY',true)+rect(246,74,216,34,'m-outline','fill="none"')+txt(256,88,'calibration + corner YAML',true)+txt(256,101,'SHA-256 snapshot verified',true);
  const ready=between(p,.55,.93);
  body+=rect(246,119,102,72,'m-surface')+rect(360,119,102,72,'m-surface');
  body+=txt(255,136,'SEGMENTATION',true)+txt(370,136,'POSE',true);
  body+=path('M277 160L281 146L312 146L320 160L315 177L282 177Z','m-fill');
  const modelPts=corners(410,160,40,27,27);
  body+=cube(modelPts,12,'m-soft')+modelPts.map(point=>dot(...point,1.7)).join('');
  body+=`<g opacity="${f(ready)}">${txt(257,204,'best.pt',true)}${txt(372,204,'best.pt',true)}</g>`;
  body+=progress(p,['Pin corners','Sample draws','Two tasks','Artifacts']);
  return {body,phase:['Verify the frozen geometry snapshot','Resample backgrounds around fixed foreground','Optimize separate segmentation and pose tasks','Keep two checkpoints with their pinned geometry'][step],readout:`Illustrative repeated source draw ${draw} / 12`,caption:'Animation demonstrates background sampling and artifact roles only. It does not represent optimizer progress, loss or model quality; validation uses original photographs.'};
}
const displays=[calibration,capture,masks,exportLabels,fusion,training];
export function datasetMotion(index,p) {
  const selected=Math.max(0,Math.min(5,Math.round(Number.isFinite(Number(index))?Number(index):0)));
  return displays[selected](clamp(p));
}
