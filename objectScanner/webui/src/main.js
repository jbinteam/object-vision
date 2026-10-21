import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { frameMatrix, frameFromPicks, frameFromPose, cropGeometry, moveCropFace, cropSliderBounds, cropRecipe, inside, validCrop, matrixRows } from './geometry.js';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const requestedFrame = new URLSearchParams(location.search).get('frame_id');
const objectFrame = requestedFrame && /^[A-Za-z][A-Za-z0-9_-]{0,63}$/.test(requestedFrame)
  ? requestedFrame : 'obj_frame';
const frameLocked = new URLSearchParams(location.search).get('lock_frame')==='1';
const cropOnly = new URLSearchParams(location.search).get('crop_only')==='1';
const state = { cloud: null, min: [0,0,0], max: [1,1,1], cropCenter:[.5,.5,.5], cropRotation:[0,0,0], cropTool:'off', origin: [0,0,0], rotation: [0,0,0], mode:'scan', tab:'crop', frameTool:'translate', picks: null, busy:false };
let scene, camera, renderer, controls, root, points, cropBox, objectAxes, scanAxes, material, renderPending=false;
let frameHandle, cropHandle, frameControls;
let noticeTimer, countTimer, countRequest=0, sourceVersion=0;
let autoCenterOrigin=objectFrame==='box_center';

function notice(message, error=false) {
  const element = $('#notice');
  element.textContent = message; element.classList.toggle('error', error); element.hidden=false;
  clearTimeout(noticeTimer); noticeTimer=setTimeout(()=>element.hidden=true, error ? 10000 : 5000);
}

async function api(path, options={}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { message = (await response.json()).error || message; } catch {}
    throw new Error(message);
  }
  return response;
}
const post = async (path, data) => (await api(path, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})).json();

async function busy(label, action) {
  if(state.busy) return;
  state.busy=true; $('#busy-label').textContent=label; $('#busy').hidden=false;
  try { return await action(); } catch (error) { notice(error.message, true); }
  finally { state.busy=false; $('#busy').hidden=true; }
}

function render() {
  if(!renderer || renderPending) return;
  renderPending=true;
  requestAnimationFrame(()=>{renderPending=false;renderer.render(scene,camera);});
}

function initScene() {
  if(renderer) return;
  const target=$('#viewport');
  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(45,1,0.0001,10000);
  camera.up.set(0,-1,0);
  renderer=new THREE.WebGLRenderer({alpha:true,antialias:false,powerPreference:'low-power'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  target.prepend(renderer.domElement);
  renderer.domElement.setAttribute('aria-label','Interactive point cloud. Drag to orbit, right-drag to pan, scroll to zoom.');
  renderer.domElement.addEventListener('webglcontextlost', event => {
    event.preventDefault(); notice('The browser lost its 3D graphics context. Reload this page to reopen the scan.',true);
  });
  controls=new OrbitControls(camera, renderer.domElement);
  controls.enableDamping=false;
  controls.addEventListener('change',render);
  root=new THREE.Group(); root.matrixAutoUpdate=false; scene.add(root);
  frameHandle=new THREE.Object3D();root.add(frameHandle);
  cropHandle=new THREE.Group();root.add(cropHandle);
  frameControls=new TransformControls(camera,renderer.domElement);
  frameControls.setSpace('local');frameControls.setSize(.85);
  scene.add(frameControls.getHelper());
  frameControls.addEventListener('change',render);
  frameControls.addEventListener('dragging-changed',event=>{
    controls.enabled=!event.value && !state.picks;
    renderer.domElement.style.cursor=event.value?'grabbing':'';
  });
  frameControls.addEventListener('objectChange',()=>{
    const target=frameControls.object;
    const frame=frameFromPose(target.position,target.quaternion);
    if(target===cropHandle){state.cropRotation=frame.rotation;syncCropInputs();}
    else {autoCenterOrigin=false;state.origin=frame.origin;state.rotation=frame.rotation;syncFrameInputs();}
    updateAll();
  });
  renderer.domElement.addEventListener('pointercancel',()=>frameControls.pointerUp(null));
  syncFrameTool();
  new ResizeObserver(()=>{
    const w=target.clientWidth,h=target.clientHeight;
    renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();render();
  }).observe(target);
  let down;
  renderer.domElement.addEventListener('pointerdown',event=>down=[event.clientX,event.clientY]);
  renderer.domElement.addEventListener('pointerup',event=>{
    if(state.picks && down && Math.hypot(event.clientX-down[0],event.clientY-down[1])<5) pickPoint(event);
  });
}

function disposeObject(object) {
  if(!object) return;
  object.traverse(child=>{
    child.geometry?.dispose();
    if(Array.isArray(child.material))child.material.forEach(m=>m.dispose());else child.material?.dispose();
  });
  object.removeFromParent();
}

function installCloud(metadata, buffer) {
  initScene();
  [points,cropBox,objectAxes,scanAxes].forEach(disposeObject);
  const data=new Float32Array(buffer);
  const positions=new Float32Array(data.length/2),colors=new Float32Array(data.length/2);
  for(let i=0,j=0;i<data.length;i+=6,j+=3){positions.set(data.subarray(i,i+3),j);colors.set(data.subarray(i+3,i+6),j);}
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
  geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
  geometry.computeBoundingSphere();
  material=new THREE.ShaderMaterial({
    uniforms:{cropMin:{value:new THREE.Vector3()},cropMax:{value:new THREE.Vector3()},scanToCrop:{value:new THREE.Matrix4()},pointSize:{value:Number($('#point-size').value)*renderer.getPixelRatio()},showOutside:{value:true}},
    vertexShader:`attribute vec3 color; varying vec3 vColor; varying float vInside; uniform vec3 cropMin; uniform vec3 cropMax; uniform mat4 scanToCrop; uniform float pointSize;
      void main(){vec3 local=(scanToCrop*vec4(position,1.0)).xyz;vInside=all(greaterThanEqual(local,cropMin))&&all(lessThanEqual(local,cropMax))?1.0:0.0;vColor=color;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=pointSize;}`,
    fragmentShader:`varying vec3 vColor; varying float vInside; uniform bool showOutside;
      void main(){if(vInside<0.5&&!showOutside)discard;if(length(gl_PointCoord-0.5)>0.5)discard;gl_FragColor=vec4(vInside>0.5?vColor:vec3(0.15,0.19,0.23),1.0);}`
  });
  points=new THREE.Points(geometry,material);root.add(points);
  cropBox=new THREE.Box3Helper(new THREE.Box3(),0xe8bd7f);cropHandle.add(cropBox);
  objectAxes=new THREE.AxesHelper(1);objectAxes.matrixAutoUpdate=false;objectAxes.material.depthTest=false;objectAxes.renderOrder=2;root.add(objectAxes);
  scanAxes=new THREE.AxesHelper(Math.max(...metadata.extent)*0.12);scanAxes.material.transparent=true;scanAxes.material.opacity=.25;root.add(scanAxes);
  const suggestedCrop=metadata.suggested_crop;
  state.cloud=metadata;state.min=suggestedCrop?[...suggestedCrop.crop_min]:[...metadata.bounds_min];state.max=suggestedCrop?[...suggestedCrop.crop_max]:[...metadata.bounds_max];state.rotation=[0,0,0];state.mode='scan';
  state.cropRotation=suggestedCrop?[...suggestedCrop.crop_rotation_deg]:[0,0,0];
  state.cropCenter=state.min.map((value,i)=>(value+state.max[i])/2);
  const suggested=metadata.suggested_frame;
  autoCenterOrigin=!suggested&&objectFrame==='box_center';
  state.origin=suggested?[...suggested.origin]:(autoCenterOrigin?[...state.cropCenter]:[0,0,0]);
  if(suggested)state.rotation=[...suggested.rotation_deg];
  sourceVersion++;countRequest++;cancelPick();
  $('#source-title').textContent=metadata.name;
  $('#empty-state').hidden=true;$('#viewport-tag').hidden=false;$('#view-legend').hidden=false;
  $$('fieldset,.requires-cloud,#reset-crop,#reset-frame').forEach(element=>element.disabled=false);
  $('#total-count').textContent=metadata.points.toLocaleString();
  $('#preview-count').textContent=metadata.preview_points.toLocaleString();
  const requestedName=new URLSearchParams(location.search).get('reference_name');
  $('#reference-name').value=requestedName&&/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(requestedName)
    ? requestedName
    : metadata.name.replace(/\.ply$/i,'').replace(/[^a-zA-Z0-9_-]/g,'_').replace(/^[^a-zA-Z0-9]+/,'').slice(0,54)+'_ref';
  $('#export-result').hidden=true;
  buildCropInputs();syncFrameInputs();updateAll();setView('scan');fit();setTab('crop');
  if(frameLocked){
    $$('#origin-inputs input,#rotation-inputs input,#reset-frame,#center-frame,#pick-frame,[data-frame-tool]').forEach(element=>element.disabled=true);
    $('#panel-frame .hint').textContent='The object frame is locked to the AprilTag calibration so PnP and the reference use the same axes. Move the crop center in the Crop tab.';
    notice(`Frame locked to ${suggested?.frame_id||objectFrame} from capture calibration. Crop only; the axes are shared with image labels.`);
  }
  if(cropOnly){
    $('#reference-name').disabled=true;
    $('#export-voxel').disabled=true;
    $('#export-normals').disabled=true;
  }
}

async function openMetadata(metadata){
  const buffer=await (await api(`/api/clouds/${metadata.id}/preview`)).arrayBuffer();
  installCloud(metadata,buffer);
  notice(`${metadata.points.toLocaleString()} points loaded. Coordinates shown in metres.`);
}

function buildCropInputs() {
  const container=$('#crop-axes');container.replaceChildren();
  ['X','Y','Z'].forEach((axis,i)=>{
    const section=document.createElement('div');section.className='axis-block';
    section.innerHTML=`<div class="axis-title"><b class="${axis.toLowerCase()}">${axis}</b><span>${['Width','Height','Depth'][i]}</span></div><div class="axis-values"><label>MIN<input type="number" step="0.001" aria-label="${axis} crop minimum" data-bound="min"></label><label>MAX<input type="number" step="0.001" aria-label="${axis} crop maximum" data-bound="max"></label></div><div class="range-pair"><input type="range" step="0.0001" aria-label="${axis} minimum slider" data-bound="min"><input type="range" step="0.0001" aria-label="${axis} maximum slider" data-bound="max"></div>`;
    section.querySelectorAll('input').forEach(input=>{
      input.dataset.axis=i;
      if(input.type==='range'){input.min=state.cloud.bounds_min[i];input.max=state.cloud.bounds_max[i];}
      input.addEventListener(input.type==='range'?'input':'change',()=>{
        const value=Number(input.value),side=input.dataset.bound;
        if(!Number.isFinite(value) || (side==='min' ? value>=state.max[i] : value<=state.min[i])){
          notice('Minimum must be below maximum on every axis.',true);syncCropInputs();return;
        }
        const crop=moveCropFace(state.min,state.max,state.cropRotation,state.cropCenter,side,i,value);
        state.min=crop.min;state.max=crop.max;state.cropCenter=crop.center;
        if(autoCenterOrigin){state.origin=[...state.cropCenter];syncFrameInputs();}
        syncCropInputs();updateAll();
      });
    });container.append(section);
  });syncCropInputs();
}
function syncCropInputs(){
  const bounds=state.cloud?cropSliderBounds(state.cloud.bounds_min,state.cloud.bounds_max,state.min,state.max,state.cropRotation,state.cropCenter):null;
  $$('#crop-axes input').forEach(input=>{
    const i=Number(input.dataset.axis),v=state[input.dataset.bound][i];
    if(input.type==='range' && bounds){input.min=Math.min(bounds.min[i],state.min[i]);input.max=Math.max(bounds.max[i],state.max[i]);}
    input.value=input.type==='range'?v:Number(v.toFixed(6));
  });
  $$('#crop-rotation-inputs input').forEach(input=>input.value=Number(state.cropRotation[Number(input.dataset.axis)].toFixed(6)));
  $$('#crop-center-inputs input').forEach(input=>input.value=Number(state.cropCenter[Number(input.dataset.axis)].toFixed(6)));
}
function createCropCenterInputs(){
  const container=$('#crop-center-inputs');
  ['X','Y','Z'].forEach((axis,i)=>{
    const label=document.createElement('label');label.textContent=axis;
    const input=document.createElement('input');input.type='number';input.step='0.001';
    input.dataset.axis=i;input.setAttribute('aria-label',`${axis} crop center`);
    input.addEventListener('change',()=>{
      const value=Number(input.value);
      if(!Number.isFinite(value)){notice('Crop center must be finite.',true);syncCropInputs();return;}
      state.cropCenter[i]=value;
      if(!frameLocked&&autoCenterOrigin){state.origin=[...state.cropCenter];syncFrameInputs();}
      syncCropInputs();updateAll();
    });
    label.append(input);container.append(label);
  });
}
function createFrameInputs(){
  for(const [kind,labels] of [['origin',['X','Y','Z']],['rotation',['Roll X','Pitch Y','Yaw Z']],['cropRotation',['Roll X','Pitch Y','Yaw Z']]]){
    const container=$(kind==='cropRotation'?'#crop-rotation-inputs':`#${kind}-inputs`);
    labels.forEach((label,i)=>{
      const el=document.createElement('label');el.textContent=label;
      const input=document.createElement('input');input.type='number';input.step=kind==='origin'?'0.001':'1';input.value='0';input.setAttribute('aria-label',`${kind} ${label}`);input.dataset.axis=i;el.append(input);container.append(el);
      input.addEventListener('change',()=>{const v=Number(input.value);if(!Number.isFinite(v)){syncFrameInputs();syncCropInputs();return;}if(kind==='origin')autoCenterOrigin=false;state[kind][i]=v;if(kind==='cropRotation')syncCropInputs();cancelPick();updateAll();});
    });
  }
}
function syncFrameInputs(){for(const kind of ['origin','rotation'])$$(`#${kind}-inputs input`).forEach(input=>input.value=Number(state[kind][Number(input.dataset.axis)].toFixed(6)));}

function updateAll(){
  if(!state.cloud)return;
  const matrix=frameMatrix(state.origin,state.rotation);
  const size=Math.max(...state.max.map((x,i)=>x-state.min[i]));
  const crop=cropGeometry(state.min,state.max,state.cropRotation,state.cropCenter);
  material.uniforms.cropMin.value.copy(crop.half).negate();material.uniforms.cropMax.value.copy(crop.half);
  material.uniforms.scanToCrop.value.copy(crop.inverse);
  material.uniforms.showOutside.value=$('#show-outside').checked;
  cropBox.box.set(crop.half.clone().negate(),crop.half);
  if(!frameControls.dragging || frameControls.object!==cropHandle)crop.matrix.decompose(cropHandle.position,cropHandle.quaternion,cropHandle.scale);
  objectAxes.matrix.copy(matrix).scale(new THREE.Vector3().setScalar(Math.max(size*.25,.001)));
  if(!frameControls.dragging || frameControls.object!==frameHandle)matrix.decompose(frameHandle.position,frameHandle.quaternion,frameHandle.scale);
  root.matrix.copy(state.mode==='object'?matrix.clone().invert():new THREE.Matrix4());root.updateMatrixWorld(true);
  $('#matrix-preview').textContent=matrixRows(matrix).map(row=>row.map(v=>(Math.abs(v)<0.00005?0:v).toFixed(4).padStart(8)).join(' ')).join('\n');
  $('#frame-summary').textContent=`Origin: ${state.origin.map(v=>v.toFixed(3)).join(', ')} m`;
  $('#extent').textContent=state.max.map((v,i)=>(v-state.min[i]).toFixed(3)).join(' × ');
  $('#export-result').hidden=true;
  clearTimeout(countTimer);const version=sourceVersion;
  countTimer=setTimeout(()=>refreshCount(version),250);
  render();
}
async function refreshCount(version){
  const requestID=++countRequest;
  try{const data=await post('/api/count',payload());if(version===sourceVersion&&requestID===countRequest)$('#crop-count').textContent=data.points.toLocaleString();}
  catch(error){if(version===sourceVersion)notice(error.message,true);}
}
function payload(){return{id:state.cloud?.id,...cropRecipe(state.min,state.max,state.cropRotation,state.cropCenter),origin:[...state.origin],rotation_deg:[...state.rotation]};}

function setView(mode){
  state.mode=mode;
  $$('#view-modes button').forEach(button=>button.classList.toggle('active',button.dataset.view===mode));
  $('#frame-label').textContent=mode==='object'?objectFrame:'scan_frame';
  syncFrameTool();
  if(!state.cloud)return;
  camera.up.set(0,mode==='object'?0:-1,mode==='object'?1:0);
  updateAll();fit();
}
function fit(direction='front'){
  if(!state.cloud)return;
  const crop=cropGeometry(state.min,state.max,state.cropRotation,state.cropCenter);
  const box=new THREE.Box3(crop.half.clone().negate(),crop.half).applyMatrix4(root.matrix.clone().multiply(crop.matrix));
  const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
  const radius=Math.max(size.length()/2,.001),dist=radius/Math.sin(THREE.MathUtils.degToRad(camera.fov/2))/Math.min(camera.aspect,1)*1.1;
  const directions=state.mode==='object'?{front:[0,-1,.15],top:[0,-.001,1],side:[1,0,.1]}:{front:[0,0,-1],top:[0,-1,-.001],side:[1,0,0]};
  camera.position.copy(center).addScaledVector(new THREE.Vector3(...directions[direction]).normalize(),dist);
  camera.near=Math.max(radius/1000,1e-6);camera.far=Math.max(dist*100,100);camera.updateProjectionMatrix();
  controls.target.copy(center);controls.update();render();
}
function setTab(tab){
  state.tab=tab;
  $$('.step-tabs button').forEach(button=>button.classList.toggle('active',button.dataset.tab===tab));
  for(const name of ['crop','frame','export'])$(`#panel-${name}`).hidden=name!==tab;
  if(tab!=='frame')cancelPick();
  syncFrameTool();
}
function syncFrameTool(){
  const cropActive=state.tab==='crop' && state.cropTool==='rotate';
  const active=!!state.cloud && state.mode==='scan' && !state.picks && (cropActive || (!frameLocked&&state.tab==='frame' && state.frameTool!=='off'));
  $$('[data-crop-tool]').forEach(button=>{
    const selected=button.dataset.cropTool===state.cropTool;
    button.classList.toggle('active',selected);button.setAttribute('aria-pressed',String(selected));
  });
  $$('[data-frame-tool]').forEach(button=>{
    const selected=button.dataset.frameTool===state.frameTool;
    button.classList.toggle('active',selected);button.setAttribute('aria-pressed',String(selected));
  });
  $('#frame-drag-hint').textContent=state.mode==='object'
    ? 'Select Move or Rotate to edit the frame in Scan frame view.'
    : 'Move: drag an arrow along its axis, a square within a plane, or the center freely. Rotate: drag a ring. Esc cancels a drag.';
  $('#view-help').textContent=active
    ? (cropActive?'Drag rings to rotate crop box · Drag elsewhere to orbit':'Drag frame handles to edit · Drag elsewhere to orbit · Scroll to zoom')
    : 'Drag to orbit · Right-drag to pan · Scroll to zoom';
  if(!frameControls)return;
  frameControls.enabled=active;
  if(active){
    frameControls.setMode(cropActive?'rotate':state.frameTool);
    const target=cropActive?cropHandle:frameHandle;
    if(frameControls.object!==target)frameControls.attach(target);
  }else frameControls.detach();
  render();
}
function setFrameTool(tool){
  cancelPick();state.frameTool=tool;
  if(tool!=='off' && state.mode!=='scan')setView('scan');
  syncFrameTool();
}
function cancelPick(){state.picks=null;$('#pick-banner').hidden=true;if(controls)controls.enabled=!frameControls.dragging;syncFrameTool();}
function startPick(){
  if(!state.cloud)return;setView('scan');state.picks=[];controls.enabled=false;
  syncFrameTool();
  $('#pick-banner').hidden=false;$('#pick-instruction').textContent='1 / 3 — Click the object origin.';
}
function pickPoint(event){
  const rect=renderer.domElement.getBoundingClientRect(),mouse=new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
  const ray=new THREE.Raycaster();ray.params.Points.threshold=camera.position.distanceTo(controls.target)*.008;ray.setFromCamera(mouse,camera);
  const attribute=points.geometry.getAttribute('position');
  const crop=cropGeometry(state.min,state.max,state.cropRotation,state.cropCenter),localPoint=new THREE.Vector3();
  const hit=ray.intersectObject(points).find(hit=>inside(localPoint.fromBufferAttribute(attribute,hit.index).applyMatrix4(crop.inverse).toArray(),crop.half.clone().negate().toArray(),crop.half.toArray()));
  if(!hit){notice('Click a retained point on the object.',true);return;}
  state.picks.push([attribute.getX(hit.index),attribute.getY(hit.index),attribute.getZ(hit.index)]);
  if(state.picks.length<3){$('#pick-instruction').textContent=state.picks.length===1?'2 / 3 — Click a point along the object’s +X axis.':'3 / 3 — Click a point toward +Y (not on the X line).';return;}
  try{const frame=frameFromPicks(...state.picks);autoCenterOrigin=false;state.origin=frame.origin;state.rotation=frame.rotation;syncFrameInputs();updateAll();notice('Object frame assigned. Switch to Object frame to inspect it.');}
  catch(error){notice(error.message,true);}finally{cancelPick();}
}

async function refreshScans(){
  try{
    const items=await (await api('/api/scans')).json(),select=$('#saved-scan'),previous=select.value;
    select.replaceChildren(new Option(items.length?'Choose from output/…':'No saved scans — open a PLY file',''));
    for(const item of items)select.add(new Option(`${item.name} · ${(item.bytes/1024/1024).toFixed(1)} MB`,item.name));
    if(items.some(item=>item.name===previous))select.value=previous;
  }catch(error){notice(error.message,true);}
}

function showExport(result){
  const box=$('#export-result');box.replaceChildren();
  const title=document.createElement('strong');title.textContent=`${result.name} saved · ${result.points.toLocaleString()} points`;box.append(title);
  for(const [kind,label] of [['bundle','Download reference bundle ↗'],['ply','Download reference.ply'],['frame','Download frame.json']]){
    const a=document.createElement('a');a.href=result.downloads[kind];a.textContent=label;a.download='';box.append(a);
  }
  const location=document.createElement('p');location.className='micro';location.textContent=result.directory;box.append(location);box.hidden=false;
  if(result.backup){const backup=document.createElement('p');backup.className='micro';backup.textContent=`Original saved at ${result.backup}`;box.append(backup);}
}
async function exportReference(){
  if(!state.cloud)throw new Error('Open a cloud first.');
  const name=$('#reference-name').value.trim(),voxel=Number($('#export-voxel').value);
  if(!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name))throw new Error('Use letters, numbers, underscores or hyphens for the reference name.');
  if(!Number.isFinite(voxel)||voxel<0)throw new Error('Enter a nonnegative voxel size.');
  const result=await post('/api/export',{...payload(),name,voxel_mm:voxel,estimate_normals:$('#export-normals').checked});
  if(cropOnly)await openMetadata(await post('/api/load',{name:'reference.ply',units:'m'}));
  showExport(result);return result;
}

createFrameInputs();createCropCenterInputs();refreshScans();
if(cropOnly){
  $('#open-file').hidden=true;$('#empty-open').hidden=true;
  $('.source-panel').hidden=true;
  $('#source-units').disabled=true;
  $('[data-tab="frame"]').hidden=true;
  const cropNext=$('[data-next="frame"]');
  cropNext.dataset.next='export';
  cropNext.innerHTML='Review crop <span>→</span>';
  $('#panel-export .hint').textContent='Save only the points inside the crop. Coordinates and the PnP reference frame stay fixed; the original bundle is backed up.';
  $('#export-reference').innerHTML='Save cropped reference <span>↗</span>';
  $('.inspector-footer').textContent='CALIBRATED REFERENCE → CROP → SAVE';
  busy('Opening calibrated reference…',async()=>openMetadata(await post('/api/load',{name:'reference.ply',units:'m'})));
}
$('#open-file').onclick=$('#empty-open').onclick=()=>$('#file-input').click();
$('#file-input').onchange=event=>{const file=event.target.files[0];if(file)busy('Opening point cloud…',async()=>{const form=new FormData();form.append('file',file);form.append('units',$('#source-units').value);await openMetadata(await (await api('/api/upload',{method:'POST',body:form})).json());});event.target.value='';};
$('#load-scan').onclick=()=>{const name=$('#saved-scan').value;if(!name){notice('Choose a saved scan first.');return;}busy('Opening saved scan…',async()=>openMetadata(await post('/api/load',{name,units:$('#source-units').value})));};
$('#refresh-scans').onclick=refreshScans;
$$('[data-tab]').forEach(button=>button.onclick=()=>setTab(button.dataset.tab));
$$('[data-next]').forEach(button=>button.onclick=()=>setTab(button.dataset.next));
$$('[data-view]').forEach(button=>button.onclick=()=>{cancelPick();setView(button.dataset.view);});
$$('[data-camera]').forEach(button=>button.onclick=()=>fit(button.dataset.camera));
$('#fit-view').onclick=()=>fit();
$('#reset-crop').onclick=()=>{state.min=[...state.cloud.bounds_min];state.max=[...state.cloud.bounds_max];state.cropCenter=state.min.map((value,i)=>(value+state.max[i])/2);state.cropRotation=[0,0,0];if(autoCenterOrigin){state.origin=[...state.cropCenter];syncFrameInputs();}syncCropInputs();updateAll();};
$('#reset-crop-rotation').onclick=()=>{state.cropRotation=[0,0,0];syncCropInputs();updateAll();};
$('#center-crop').onclick=()=>{state.cropCenter=[...state.origin];syncCropInputs();updateAll();};
$$('[data-crop-tool]').forEach(button=>button.onclick=()=>{cancelPick();state.cropTool=button.dataset.cropTool;if(state.cropTool!=='off' && state.mode!=='scan')setView('scan');syncFrameTool();});
$('#reset-frame').onclick=()=>{cancelPick();autoCenterOrigin=false;state.origin=[0,0,0];state.rotation=[0,0,0];syncFrameInputs();updateAll();};
$('#center-frame').onclick=()=>{cancelPick();autoCenterOrigin=true;state.origin=[...state.cropCenter];syncFrameInputs();updateAll();};
$('#show-outside').onchange=updateAll;
$('#point-size').oninput=()=>{if(material){material.uniforms.pointSize.value=Number($('#point-size').value)*renderer.getPixelRatio();render();}};
$('#pick-frame').onclick=startPick;$('#cancel-pick').onclick=cancelPick;
$$('[data-frame-tool]').forEach(button=>button.onclick=()=>setFrameTool(button.dataset.frameTool));
$('#export-reference').onclick=()=>busy('Preparing full-resolution reference…',exportReference);
document.addEventListener('keydown',event=>{
  if(event.key!=='Escape')return;
  if(frameControls?.dragging){frameControls.reset();frameControls.pointerUp(null);}
  cancelPick();
});

// Optional browser-native agent tools share the exact same editing actions.
if(document.modelContext?.registerTool){
  const tools=[{
    name:'read_object_reference',description:'Read the loaded cloud, crop and object frame.',inputSchema:{type:'object',properties:{}},annotations:{readOnlyHint:true},
    execute:async()=>({cloud:state.cloud,...payload()})
  },{
    name:'set_object_crop',description:'Set crop bounds before rotation in scan metres. Optional rotation_deg is roll, pitch, yaw around the box midpoint; omitted rotation preserves the current orientation.',
    inputSchema:{type:'object',properties:{minimum:{type:'array',items:{type:'number'},minItems:3,maxItems:3},maximum:{type:'array',items:{type:'number'},minItems:3,maxItems:3},rotation_deg:{type:'array',items:{type:'number'},minItems:3,maxItems:3}},required:['minimum','maximum']},annotations:{readOnlyHint:false},
    execute:async input=>{if(!state.cloud||!validCrop(input.minimum,input.maximum))throw new Error('Load a cloud and supply valid crop bounds.');if(input.rotation_deg && (input.rotation_deg.length!==3 || !input.rotation_deg.every(Number.isFinite)))throw new Error('Supply three finite rotation angles.');state.min=[...input.minimum];state.max=[...input.maximum];state.cropCenter=state.min.map((value,i)=>(value+state.max[i])/2);if(autoCenterOrigin){state.origin=[...state.cropCenter];syncFrameInputs();}if(input.rotation_deg)state.cropRotation=[...input.rotation_deg];syncCropInputs();updateAll();return payload();}
  }];
  for(const tool of tools)Promise.resolve(document.modelContext.registerTool(tool)).catch(()=>{});
}
