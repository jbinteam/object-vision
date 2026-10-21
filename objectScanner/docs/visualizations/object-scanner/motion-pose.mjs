// Deterministic explanatory screen data, not a camera replay or solver output.
// Sources: orbbec_camera.py:298–381; camera.py:125–134;
// ros2_ws/src/object_pose_tracking/object_pose_tracking/
// {observation_node.py:75–126,segmentation.py:22–72,geometry.py:28–102,
// pnp.py:103–204,gicp.py:79–204,pnp_gicp_node.py:187–280,
// pose_filter.py:8–65}; config/phase4.yaml:1–110.

const clamp = value => Math.max(0, Math.min(1, Number.isFinite(Number(value)) ? Number(value) : 0));
const mix = (a,b,t) => a+(b-a)*t;
const ramp = (p,a,b) => clamp((p-a)/(b-a));
const ease = t => t*t*(3-2*t);
const n = value => Number(value.toFixed(2));
const esc = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const text = (x,y,value,cls='m-small',extra='') => `<text x="${n(x)}" y="${n(y)}" class="${cls}" ${extra}>${esc(value)}</text>`;
const rect = (x,y,w,h,cls='m-surface',r=5) => `<rect x="${n(x)}" y="${n(y)}" width="${n(w)}" height="${n(h)}" rx="${r}" class="${cls}"/>`;
const dot = (x,y,r=2,cls='m-ink',opacity=1) => `<circle cx="${n(x)}" cy="${n(y)}" r="${r}" class="${cls}" opacity="${n(opacity)}"/>`;
const line = (x1,y1,x2,y2,cls='m-soft',extra='') => `<path d="M${n(x1)} ${n(y1)}L${n(x2)} ${n(y2)}" class="${cls}" fill="none" ${extra}/>`;
const cross = (x,y,r=4,cls='m-dim') => line(x-r,y-r,x+r,y+r,cls)+line(x-r,y+r,x+r,y-r,cls);
const poly = (points,cls='m-outline',closed=false,extra='') => `<path d="${points.map(([x,y],i)=>(i?'L':'M')+n(x)+' '+n(y)).join('')}${closed?'Z':''}" class="${cls}" fill="none" ${extra}/>`;
const arrow = (a,b,cls='m-soft') => {
  const angle=Math.atan2(b[1]-a[1],b[0]-a[0]),l=5;
  return line(...a,...b,cls)+poly([[b[0]-l*Math.cos(angle-.55),b[1]-l*Math.sin(angle-.55)],b,[b[0]-l*Math.cos(angle+.55),b[1]-l*Math.sin(angle+.55)]],cls);
};
const heading = value => text(24,27,value,'m-label');
const footer = value => {
  if(value.length<=76)return text(24,242,value,'m-small');
  const split=value.lastIndexOf(' ',76);
  return text(24,230,value.slice(0,split),'m-small')+text(24,245,value.slice(split+1),'m-small');
};
const objectPoints = [[0,0],[42,0],[42,29],[29,29],[29,48],[0,48]];
const object = (x,y,s=1,cls='m-outline') => poly(objectPoints.map(([a,b])=>[x+a*s,y+b*s]),cls,true);
const cube = (x,y,s=1) => [[x,y+16*s],[x+45*s,y+16*s],[x,y+62*s],[x+45*s,y+62*s],[x+16*s,y],[x+61*s,y],[x+16*s,y+46*s],[x+61*s,y+46*s]];
const cubeEdges = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
const drawCube = (points,cls='m-soft') => cubeEdges.map(([a,b])=>line(...points[a],...points[b],cls)).join('');
const phaseResult = (body,phase,readout,caption) => ({body,phase,readout,caption});

function alignment(p) {
  const a=ease(ramp(p,.08,.76)),shift=18*(1-a);
  let body=heading('RECTIFIED PIXELS · ONE PACKET');
  for(const [x,label] of [[24,'RGB'],[254,'ALIGNED DEPTH']]) {
    body+=rect(x,44,202,148)+text(x+12,63,label,'m-label');
    body+=line(x+18,174,x+184,174,'m-dim')+line(x+101,74,x+101,181,'m-dim');
  }
  body+=object(97,90,1.15,'m-ink')+object(327+shift,90-shift*.45,1.15,'m-ink');
  const sweep=77+102*p;
  body+=line(40,sweep,210,sweep,'m-soft')+line(270,sweep,440,sweep,'m-soft');
  for(let row=0;row<5;row++)for(let col=0;col<4;col++){
    if(row>2&&col>1)continue;
    body+=dot(333+shift+col*10,96-shift*.45+row*10,1.7,'m-ink',.4+.6*a);
  }
  body+=text(37,184,'rectify color')+text(267,184,'500 mm → 0.500 m');
  body+=line(125,200,355,200,'m-dim');
  body+=dot(mix(125,240,a),200,3,'m-lit')+dot(mix(355,240,a),200,3,'m-lit');
  body+=text(240,220,a>.98?'same stamp · same optical frame':'match geometry + packet stamp','m-small','text-anchor="middle"');
  body+=footer('Alignment is geometric; a shared stamp does not prove simultaneous exposure.');
  return phaseResult(body,p<.25?'Receive RGB + depth':p<.76?'Register and rectify':'Publish one coherent packet',a>.98?'RGB8 + 16UC1 mm + current K':'Illustrative alignment sequence','Depth and color converge onto one rectified pixel grid. Real registration uses camera calibration; this schematic shift is not the SDK algorithm.');
}

function inference(p) {
  const selected=p>=.18,paired=p>=.38&&p<.68,tracking=p>=.68;
  let body=heading('MASK + INSTANCE ASSOCIATION');
  body+=rect(24,44,279,159)+rect(318,44,138,159);
  body+=object(89,83,1.8,selected?'m-lit':'m-soft')+object(226,117,.7,'m-dim');
  body+=rect(77,70,104,108,selected?'m-outline':'m-dim',1)+rect(219,108,43,54,'m-dim',1);
  if(selected)body+=text(83,65,'selected mask','m-small');
  if(paired){
    const pts=cube(85,75,1.25);
    body+=drawCube(pts,'m-soft');
    pts.forEach(([x,y],i)=>{body+=dot(x,y,2.8,'m-lit')+text(x+4,y-4,i,'m-small');});
  }
  const stripe=76+94*((p*.85)%1);
  if(selected)body+=line(80,stripe,178,stripe,'m-soft');
  body+=text(329,65,tracking?'TRACKING':'INITIALIZE','m-label');
  body+=text(329,93,'segmentation: on');
  body+=text(329,115,tracking?'corners: skipped':'corners: needed');
  body+=text(329,146,paired?'pair IoU = 0.62':tracking?'raw pose → seed':'best box overlap');
  body+=text(329,173,paired?'other IoU = 0.08':tracking?'mask continues':'minimum = 0.10');
  body+=footer(tracking?'TRACKING keeps segmenting; PnP corners return after counted loss.':'Corner IDs keep their trained order; pairing selects one matching instance.');
  return phaseResult(body,!selected?'Detect target class':paired?'Pair ordered corners':tracking?'Skip pose network':'Select segmentation instance',tracking?'Mask on · corner inference off':paired?'0.62 ≥ 0.10 · pair selected':'Segmentation confidence threshold 0.50','Illustrative detections show two separate models. Spatial box association does not guarantee persistent identity among similar objects.');
}

function cloud(p) {
  const travel=ease(ramp(p,.16,.57)),voxel=ease(ramp(p,.57,.79)),reject=ramp(p,.79,1);
  let body=heading('PIXELS → METRIC CLOUD → FILTERED CLOUD');
  body+=rect(24,45,167,162)+rect(253,45,203,162);
  body+=text(36,64,'masked depth','m-label')+text(265,64,'camera XYZ','m-label');
  for(let i=0;i<64;i++){
    const group=i%36,c=group%6,r=Math.floor(group/6),bad=group===0||group===5||group===30||group===35;
    const pixel=[57+(i%8)*12,88+Math.floor(i/8)*12];
    let target=[287+c*24,88+r*18];
    if(bad)target=[target[0]+(c===0?-14:14),target[1]+(r===0?-8:8)];
    const jitter=i<36?[-2,-2]:[3,3];
    const x=mix(pixel[0],target[0]+jitter[0]*(1-voxel),travel),y=mix(pixel[1],target[1]+jitter[1]*(1-voxel),travel);
    const alpha=(i>=36?1-voxel:1)*(bad?1-reject:1);
    if(alpha>.01)body+=dot(x,y,travel>.99?2.1:1.6,bad&&reject>.1?'m-dim':'m-ink',alpha);
    if(i<36&&bad&&reject>.1&&reject<.98)body+=cross(x,y,4,'m-dim');
  }
  body+=arrow([202,126],[241,126]);
  body+=text(36,198,'u,v + depth + K')+text(265,198,voxel>.99?'voxel mean · outlier gate':'X=(u−cx)Z/fx');
  body+=footer('Illustrative counts: 64 depth samples → 36 voxel means → 32 retained points.');
  const readout=p<.57?'500 mm · example X = 0.100 m':p<.79?'3 mm voxels · average occupied cells':p<1?'Remove four isolated voxel representatives':'32 retained ≥ minimum 30';
  return phaseResult(body,p<.16?'Select masked depth':p<.57?'Backproject valid pixels':p<.79?'Average within voxels':'Remove outliers',readout,'The diagram counts sample glyphs, not live camera points. A foreground pixel at u=420 with cx=320, fx=500 and Z=0.5 m backprojects to X=0.100 m.');
}

function seeds(p) {
  const fit=ease(ramp(p,.1,.42)),correction=ramp(p,.42,.66),tracking=p>=.72;
  let body=heading('INITIALIZATION OR ACCEPTED RAW HISTORY');
  body+=rect(24,45,216,163)+rect(257,45,199,163);
  if(!tracking){
    const detected=cube(81,80,1.1),estimate=cube(81+16*(1-fit),80-9*(1-fit),1.1);
    body+=drawCube(estimate,'m-soft');
    detected.forEach(([x,y],i)=>{body+=i<5?dot(x,y,3,'m-lit'):cross(x+8,y-5,3);body+=text(x+5,y+11,i,'m-small');});
    body+=text(36,65,'ordered 3D ↔ 2D','m-label');
    body+=text(36,194,fit>.99?'5 inliers · mean 2 px':'RANSAC → refine inliers');
    body+=text(269,65,'PnP → GICP seed','m-label');
    body+=text(269,96,'raw PnP Z: 0.550 m');
    body+=text(269,119,'cloud median: 0.500 m');
    body+=line(286,143,426,143,'m-dim');
    body+=dot(426-100*correction,143,4,'m-lit');
    body+=text(269,171,correction>.99?'seed Z: 0.500 m':'50 mm Z correction');
    body+=text(269,192,'X, Y, R unchanged');
  }else{
    body+=text(36,65,'TRACKING','m-label')+drawCube(cube(93,83,1),'m-ink');
    body+=text(36,194,'skip PnP for this fit');
    body+=text(269,65,'previous accepted raw','m-label');
    body+=rect(274,80,166,36,'m-outline')+text(286,103,'raw GICP pose → seed');
    body+=text(274,145,'filtered display pose');
    body+=text(274,164,'is not the next seed');
    body+=arrow([240,128],[257,128]);
    body+=text(274,192,'same box_center frame');
  }
  body+=footer(tracking?'A new accepted raw fit replaces history; rejected candidates never do.':'Depth distance gate is full XYZ; bounded correction changes only seed Z.');
  return phaseResult(body,tracking?'Reuse accepted raw pose':p<.42?'Fit corner consensus':p<.66?'Correct a bounded Z discrepancy':'Preserve raw PnP diagnostic',tracking?'TRACKING → previous raw GICP seed':correction>.99?'50 mm lies in inclusive 20–80 mm band':'Illustrative PnP depth distance 50 mm ≤ 120 mm','PnP is used at startup or recovery. Its raw published result stays unchanged when the optional GICP-seed depth correction is applied.');
}

function refinement(p) {
  const t=ease(ramp(p,.2,.78)),angle=mix(.16,.025,t),dx=mix(24,3,t),dy=mix(-12,-1,t);
  let body=heading('ONE FINE RIGID UPDATE · SCHEMATIC');
  body+=rect(24,45,270,162)+rect(310,45,146,162);
  const points=[];
  for(let j=0;j<6;j++)for(let i=0;i<7;i++)if(!(i>3&&j>3))points.push([-48+i*15,-35+j*14]);
  // One shared rotation and translation moves every source point rigidly.
  for(const [x,y]of points)body+=dot(147+x,127+y,2.4,'m-dim');
  for(const [x,y]of points){
    const xx=147+Math.cos(angle)*x-Math.sin(angle)*y+dx,yy=127+Math.sin(angle)*x+Math.cos(angle)*y+dy;
    body+=dot(xx,yy,1.8,'m-lit');
  }
  body+=line(38,189,55,189,'m-dim')+text(61,193,'observed')+dot(161,189,2,'m-lit')+text(171,193,'reference');
  body+=text(322,65,'CURRENT PRESET','m-label');
  body+=text(322,91,'fine level only')+text(322,113,'voxel: 2 mm')+text(322,135,'pairs within 8 mm')+text(322,157,'budget: 1 iteration');
  body+=text(322,188,t>.99?'then quality gates':'local rigid fit');
  body+=footer('Require ≥30 pairs, fitness ≥0.20, RMSE ≤12 mm and target overlap ≥0.10.');
  return phaseResult(body,p<.2?'Start from a local seed':p<.78?'Apply one shared rigid update':'Check fit + image/motion gates',t>.99?'Candidate still requires every relevant gate':'All reference points share the same R and t','This deterministic rigid interpolation illustrates local registration; it is not a GICP implementation, convergence trace or accuracy measurement. Current runtime disables coarse and medium levels.');
}

function publication(p) {
  const i=Math.min(5,Math.floor(p*6));
  const events=[
    {label:'prior',state:'TRACKING',raw:'0.100 m',filtered:'0.100 m',fail:'0',phase:'Prior accepted state'},
    {label:'accept',state:'TRACKING',raw:'0.120 m',filtered:'0.109 m',fail:'0',phase:'Accept and smooth'},
    {label:'reject',state:'TRACKING',raw:'0.120 m',filtered:'0.109 m',fail:'1',phase:'Reject preserves seed + filter'},
    {label:'stale',state:'TRACKING',raw:'0.120 m',filtered:'0.109 m',fail:'1',phase:'Drop stale input; no reject call'},
    {label:'reject',state:'LOST',raw:'cleared',filtered:'reset',fail:'2',phase:'Second counted failure → LOST'},
    {label:'recover',state:'TRACKING',raw:'0.130 m',filtered:'0.130 m',fail:'0',phase:'PnP recovery → accepted fit'},
  ],e=events[i];
  let body=heading('COMMIT, SMOOTH OR PRESERVE STATE');
  body+=line(48,75,432,75,'m-dim');
  events.forEach((ev,j)=>{const x=48+j*76.8;body+=dot(x,75,j===i?5:3,j===i?'m-lit':'m-dim')+text(x,98,ev.label,'m-small','text-anchor="middle"');});
  body+=rect(24,113,432,95)+text(37,134,e.state,'m-label');
  body+=text(37,159,'RAW SEED X','m-small')+text(181,159,'FILTERED X','m-small')+text(340,159,'FAILURES','m-small');
  body+=text(37,187,e.raw,'m-label')+text(181,187,e.filtered,'m-label')+text(340,187,e.fail,'m-label');
  body+=footer(i===1?'0.55 × 0.100 + 0.45 × 0.120 = 0.109 m':i===4?'Clear raw seed + reset filter; resume corner inference for recovery.':i===5?'First accepted result after reset initializes the filter unchanged.':'Only accepted poses enter the filter; silence and stale drops do not count as loss.');
  return phaseResult(body,e.phase,i===3?'Stale drop: failures stay at 1':i===4?'Two reject calls → LOST':i===1?'Next seed uses raw 0.120 m; display uses 0.109 m':`${e.state} · counted failures ${e.fail}`,'Illustrative event sequence. Rotation uses SLERP at 0.45; translation uses EMA at 0.45 per accepted update. Rejected frames show no old box, while stale early returns publish no replacement overlay.');
}

export function poseMotion(index,p) {
  const value=clamp(p),i=Math.max(0,Math.min(5,Math.round(Number.isFinite(Number(index))?Number(index):0)));
  return [alignment,inference,cloud,seeds,refinement,publication][i](value);
}
