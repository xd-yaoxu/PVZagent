#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dwmapi.h>
#include <d3d11.h>
#include <dxgi.h>
#include <roapi.h>
#include <windows.graphics.capture.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <algorithm>
#include <cstring>
#include <cwchar>
#include <limits>
#include <memory>
#include "pvz_wgc.h"
#include "pvz_wgc_alpha_diag.hpp"
#include "pvz_wgc_ts_gate.hpp"
namespace cap=winrt::Windows::Graphics::Capture;
namespace dx=winrt::Windows::Graphics::DirectX;
namespace rt=winrt::Windows::Graphics::DirectX::Direct3D11;
using Access=Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess;
namespace {
constexpr const wchar_t* allowed_exe_names[] = {L"PVZ-Windowed.exe", L"PlantsVsZombies.exe", L"popcapgame1.exe"};
bool match_pvz_exe_path(const wchar_t* full_path) {
    if (!full_path || !*full_path) return false;
    wchar_t override_path[32768]{};
    DWORD n = GetEnvironmentVariableW(L"PVZ_GAME_EXE", override_path, static_cast<DWORD>(std::size(override_path)));
    if (n > 0 && n < std::size(override_path)) return _wcsicmp(full_path, override_path) == 0;
    const wchar_t* base = full_path;
    for (const wchar_t* p = full_path; *p; ++p) { if (*p == L'\\' || *p == L'/') base = p + 1; }
    for (const wchar_t* allowed : allowed_exe_names) { if (_wcsicmp(base, allowed) == 0) return true; }
    return false;
}
constexpr uint32_t version=1;
struct Handle { HANDLE h{}; bool close(){if(!h)return true;if(!CloseHandle(h))return false;h=nullptr;return true;} ~Handle(){if(h)CloseHandle(h);} };
struct Context {
    PvzWgcRequest r{}; DWORD thread{}; Handle process; bool initialized=false, poisoned=false, cleanup_bad=false;
    uint32_t frames=0,reads=0; uint64_t opened_qpc=0, last_qpc=0; HMONITOR monitor{}; UINT dpi{};
    winrt::com_ptr<ID3D11Device> device; winrt::com_ptr<ID3D11DeviceContext> dc;
    rt::IDirect3DDevice rtdev{nullptr}; cap::GraphicsCaptureItem item{nullptr};
    cap::Direct3D11CaptureFramePool pool{nullptr}; cap::GraphicsCaptureSession session{nullptr};
    cap::Direct3D11CaptureFrame pending{nullptr}; winrt::com_ptr<ID3D11Texture2D> pending_stage;
};
uint64_t qpc100(){ LARGE_INTEGER q{},f{}; if(!QueryPerformanceCounter(&q)||!QueryPerformanceFrequency(&f)||q.QuadPart<=0||f.QuadPart<=0)return 0;
    auto a=static_cast<uint64_t>(q.QuadPart),b=static_cast<uint64_t>(f.QuadPart);if(a/b>(std::numeric_limits<uint64_t>::max)()/10000000 || a%b>(std::numeric_limits<uint64_t>::max)()/10000000)return 0;return a/b*10000000+(a%b)*10000000/b; }
bool time_ok(Context const& c){return c.r.deadline_tick_ms==0||GetTickCount64()<c.r.deadline_tick_ms;}
bool equal_rect(RECT x,int32_t l,int32_t t,int32_t r,int32_t b){return x.left==l&&x.top==t&&x.right==r&&x.bottom==b;}
bool alive(Context const& c){
    HWND w=reinterpret_cast<HWND>(static_cast<uintptr_t>(c.r.hwnd)); DWORD p{};
    DWORD cloaked{};
    if(!AreDpiAwarenessContextsEqual(GetThreadDpiAwarenessContext(),DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) || WaitForSingleObject(c.process.h,0)!=WAIT_TIMEOUT || !IsWindow(w) || !IsWindowVisible(w) || IsIconic(w) || GetForegroundWindow()!=w || !c.monitor || MonitorFromWindow(w,MONITOR_DEFAULTTONULL)!=c.monitor || GetDpiForWindow(w)!=c.dpi || GetAncestor(w,GA_ROOT)!=w || GetWindow(w,GW_OWNER)!=nullptr || FAILED(DwmGetWindowAttribute(w,DWMWA_CLOAKED,&cloaked,sizeof cloaked)) || cloaked || GetWindowThreadProcessId(w,&p)!=c.r.tid || p!=c.r.pid)return false;
    wchar_t name[128]{}; if(!GetClassNameW(w,name,128)||wcscmp(name,L"MainWindow")||wcscmp(name,c.r.class_name))return false;
    FILETIME creation{},exit{},kernel{},user{};if(!GetProcessTimes(c.process.h,&creation,&exit,&kernel,&user))return false;
    ULARGE_INTEGER n{};n.LowPart=creation.dwLowDateTime;n.HighPart=creation.dwHighDateTime;if(n.QuadPart!=c.r.creation_time_100ns)return false;
    RECT client{},dwm{};POINT a{},b{};if(!GetClientRect(w,&client)||!ClientToScreen(w,&a))return false;b={client.right,client.bottom};
    if(!ClientToScreen(w,&b)||FAILED(DwmGetWindowAttribute(w,DWMWA_EXTENDED_FRAME_BOUNDS,&dwm,sizeof dwm)))return false;
    return equal_rect({a.x,a.y,b.x,b.y},c.r.client_left,c.r.client_top,c.r.client_right,c.r.client_bottom) && equal_rect(dwm,c.r.dwm_left,c.r.dwm_top,c.r.dwm_right,c.r.dwm_bottom);
}
bool reverse(Context const& c){try {auto source=c.item.as<IWindowGraphicsCaptureItemInterop>();HWND found{};return SUCCEEDED(source->GetWindow(&found))&&found==reinterpret_cast<HWND>(static_cast<uintptr_t>(c.r.hwnd));}catch(...){return false;}}
bool cleanup(Context& c){
    if(c.poisoned)return false;bool ok=!c.cleanup_bad;
    try{if(c.session)c.session.Close();}catch(...){ok=false;}c.session=nullptr;
    try{if(c.pool)c.pool.Close();}catch(...){ok=false;}c.pool=nullptr;
    c.item=nullptr;c.rtdev=nullptr;c.dc=nullptr;c.device=nullptr;
    if(c.initialized){RoUninitialize();c.initialized=false;}return c.process.close() && ok;
}
uint32_t classify(winrt::hresult_error const& e,PvzWgcResult* out){if(out)out->hresult=static_cast<uint32_t>(e.code().value);return PVZ_API_ERROR;}
}
extern "C" int32_t __stdcall pvz_wgc_open(const PvzWgcRequest* r,void** out,PvzWgcResult* report){
    if(!out||!report||report->size!=sizeof(*report))return PVZ_DENIED;*out=nullptr;PvzWgcResult result{};result.size=sizeof result;result.open_stage=1;result.apartment_hr=result.apartment_type=result.apartment_qualifier=0xffffffffU;
    auto finish=[&](uint32_t status,uint32_t clean=0){result.status=status;result.cleanup_state=clean;*report=result;return static_cast<int32_t>(status);};
    if(!r||r->size!=sizeof(*r)||r->version!=version||!r->pid||!r->tid||!r->hwnd||!r->creation_time_100ns||!wmemchr(r->class_name,0,128)||wcscmp(r->class_name,L"MainWindow"))return finish(PVZ_DENIED);
    auto now=GetTickCount64();if(r->deadline_tick_ms!=0&&(r->deadline_tick_ms<=now||r->deadline_tick_ms-now>3600000))return finish(PVZ_DENIED);
    int64_t w=int64_t(r->client_right)-r->client_left,h=int64_t(r->client_bottom)-r->client_top;
    if(!((w==800&&h==600)||(w==1200&&h==900))||r->dwm_right<=r->dwm_left||r->dwm_bottom<=r->dwm_top)return finish(PVZ_GEOMETRY);
    if(!AreDpiAwarenessContextsEqual(GetThreadDpiAwarenessContext(),DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2))return finish(PVZ_DENIED);
    HWND target=reinterpret_cast<HWND>(static_cast<uintptr_t>(r->hwnd));DWORD pid{};wchar_t target_class[128]{};
    if(!IsWindow(target)||GetWindowThreadProcessId(target,&pid)!=r->tid||pid!=r->pid||!GetClassNameW(target,target_class,128)||wcscmp(target_class,L"MainWindow"))return finish(PVZ_DENIED);
    std::unique_ptr<Context> c;try{c=std::make_unique<Context>();}catch(...){return finish(PVZ_API_ERROR);}c->r=*r;c->thread=GetCurrentThreadId();c->monitor=MonitorFromWindow(target,MONITOR_DEFAULTTONULL);c->dpi=GetDpiForWindow(target);
    if(!c->monitor||!c->dpi)return finish(PVZ_GEOMETRY);
    result.open_stage=2;c->process.h=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION|SYNCHRONIZE,FALSE,r->pid);
    if(!c->process.h)return finish(PVZ_DENIED);
    wchar_t path[32768]{};DWORD len=static_cast<DWORD>(std::size(path));if(!QueryFullProcessImageNameW(c->process.h,0,path,&len)||!match_pvz_exe_path(path)||!alive(*c)){bool ok=cleanup(*c);return finish(ok?PVZ_DENIED:PVZ_CLEANUP_UNVERIFIED,ok?1:3);}
    try{
        result.open_stage=3;APTTYPE kind{};APTTYPEQUALIFIER qualifier{};HRESULT apartment_hr=CoGetApartmentType(&kind,&qualifier);result.apartment_hr=static_cast<uint32_t>(apartment_hr);if(SUCCEEDED(apartment_hr)){result.apartment_type=static_cast<uint32_t>(kind);result.apartment_qualifier=static_cast<uint32_t>(qualifier);}
        winrt::check_hresult(RoInitialize(RO_INIT_MULTITHREADED));c->initialized=true;c->opened_qpc=qpc100();if(!c->opened_qpc||!time_ok(*c))throw PVZ_DENIED;
        result.open_stage=4;winrt::check_hresult(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,c->device.put(),nullptr,c->dc.put()));
        result.open_stage=5;auto device=c->device.as<IDXGIDevice>();winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(device.get(),reinterpret_cast<IInspectable**>(winrt::put_abi(c->rtdev))));
        result.open_stage=6;winrt::hstring name(L"Windows.Graphics.Capture.GraphicsCaptureItem");winrt::com_ptr<IGraphicsCaptureItemInterop> factory;
        winrt::check_hresult(RoGetActivationFactory(reinterpret_cast<HSTRING>(winrt::get_abi(name)),__uuidof(IGraphicsCaptureItemInterop),factory.put_void()));
        if(!alive(*c)||!time_ok(*c))throw PVZ_DENIED;
        result.open_stage=7;winrt::check_hresult(factory->CreateForWindow(reinterpret_cast<HWND>(static_cast<uintptr_t>(r->hwnd)),__uuidof(ABI::Windows::Graphics::Capture::IGraphicsCaptureItem),winrt::put_abi(c->item)));
        result.open_stage=8;
        if(!c->item||!alive(*c)||!reverse(*c)||!time_ok(*c))throw PVZ_DENIED;
        auto size=c->item.Size();if(size.Width<=0||size.Height<=0||size.Width>8192||size.Height>8192)throw PVZ_GEOMETRY;
        result.open_stage=9;c->pool=cap::Direct3D11CaptureFramePool::CreateFreeThreaded(c->rtdev,dx::DirectXPixelFormat::B8G8R8A8UIntNormalized,1,size);
        result.open_stage=10;c->session=c->pool.CreateCaptureSession(c->item);c->session.IsCursorCaptureEnabled(false);c->session.IncludeSecondaryWindows(false);
        if(c->session.IsCursorCaptureEnabled()||c->session.IncludeSecondaryWindows()||!c->session.IsBorderRequired()||!alive(*c)||!reverse(*c)||!time_ok(*c))throw PVZ_DENIED;
        result.open_stage=11;c->session.StartCapture();result.open_stage=12;if(!alive(*c)||!reverse(*c)||!time_ok(*c))throw PVZ_DENIED;
        *out=c.release();return finish(PVZ_OK);
    }catch(winrt::hresult_error const& e){result.hresult=static_cast<uint32_t>(e.code().value);bool ok=cleanup(*c);return finish(ok?PVZ_API_ERROR:PVZ_CLEANUP_UNVERIFIED,ok?1:3);}catch(PvzWgcStatus status){bool ok=cleanup(*c);return finish(ok?status:PVZ_CLEANUP_UNVERIFIED,ok?1:3);}catch(...){bool ok=cleanup(*c);return finish(ok?PVZ_API_ERROR:PVZ_CLEANUP_UNVERIFIED,ok?1:3);}
}
extern "C" int32_t __stdcall pvz_wgc_read(void* opaque,uint8_t* output,uint64_t capacity,PvzWgcResult* out){
    if(!opaque||!output||!out||out->size!=sizeof(*out))return PVZ_DENIED;
    auto& c=*static_cast<Context*>(opaque);PvzWgcResult result{};result.size=sizeof(result);uint32_t status=PVZ_DENIED;
    if(GetCurrentThreadId()!=c.thread)return PVZ_WRONG_THREAD;if(c.poisoned)return PVZ_UNSAFE_PENDING;
    auto width=static_cast<uint32_t>(int64_t(c.r.client_right)-c.r.client_left),height=static_cast<uint32_t>(int64_t(c.r.client_bottom)-c.r.client_top);uint64_t bytes=uint64_t(width)*height*4;
    if(capacity<bytes||!time_ok(c))return PVZ_DENIED;std::memset(output,0,static_cast<size_t>(bytes));
    ++c.reads;
    cap::Direct3D11CaptureFrame frame{nullptr};
    try{
        if(!alive(c)||!reverse(c))throw PVZ_DENIED;result.identity_before=1;
        uint32_t dropped=0;
        int64_t timestamp=0;
        uint64_t now=0;
        uint32_t ts_code=PVZ_TS_OK;
        uint64_t read_wait_end=GetTickCount64()+2500; while(!frame&&time_ok(c)&&GetTickCount64()<read_wait_end){
            frame=c.pool.TryGetNextFrame();
            if(!frame){if(!alive(c)||!reverse(c))throw PVZ_DENIED;Sleep(2);continue;}
            timestamp=frame.SystemRelativeTime().count();
            now=qpc100();
            ts_code=pvz_wgc_classify_frame_ts(timestamp,c.opened_qpc,c.last_qpc,now);
            if(pvz_wgc_should_drop_buffered_frame(c.frames,dropped,ts_code,timestamp,now)){
                try{frame.Close();}catch(...){frame=nullptr;c.cleanup_bad=true;throw PVZ_CLEANUP_UNVERIFIED;}
                frame=nullptr;
                ++dropped;
                result.stale_preopen_dropped=dropped;
                if(!alive(c)||!reverse(c))throw PVZ_DENIED;
                Sleep(2);
                continue;
            }
            break;
        }
        if(!frame){result.stale_preopen_dropped=dropped;result.status=PVZ_NO_FRAME;*out=result;return PVZ_NO_FRAME;}
        ++c.frames;result.frame_index=c.frames;result.stale_preopen_dropped=dropped;
        result.ts_lead_us=pvz_wgc_compute_lead_us(timestamp,now);result.reserved0=0;
        if(ts_code!=PVZ_TS_OK){result.ts_reject_code=ts_code;throw PVZ_DENIED;}
        auto content=frame.ContentSize();auto access=frame.Surface().as<Access>();winrt::com_ptr<ID3D11Texture2D> texture;winrt::check_hresult(access->GetInterface(__uuidof(ID3D11Texture2D),texture.put_void()));
        D3D11_TEXTURE2D_DESC desc{};texture->GetDesc(&desc);result.content_w=content.Width;result.content_h=content.Height;result.texture_w=desc.Width;result.texture_h=desc.Height;
        int64_t x=int64_t(c.r.client_left)-c.r.dwm_left,y=int64_t(c.r.client_top)-c.r.dwm_top;
        if(content.Width!=int64_t(c.r.dwm_right)-c.r.dwm_left||content.Height!=int64_t(c.r.dwm_bottom)-c.r.dwm_top||x<0||y<0||x+width>content.Width||y+height>content.Height||x+width>desc.Width||y+height>desc.Height||desc.Format!=DXGI_FORMAT_B8G8R8A8_UNORM||desc.MipLevels!=1||desc.ArraySize!=1||desc.SampleDesc.Count!=1)throw PVZ_GEOMETRY;
        D3D11_TEXTURE2D_DESC dest{};dest.Width=width;dest.Height=height;dest.MipLevels=1;dest.ArraySize=1;dest.Format=desc.Format;dest.SampleDesc.Count=1;dest.Usage=D3D11_USAGE_STAGING;dest.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        winrt::com_ptr<ID3D11Texture2D> staging;winrt::check_hresult(c.device->CreateTexture2D(&dest,nullptr,staging.put()));D3D11_QUERY_DESC qd{D3D11_QUERY_EVENT,0};winrt::com_ptr<ID3D11Query> query;winrt::check_hresult(c.device->CreateQuery(&qd,query.put()));
        if(!alive(c)||!reverse(c)||!time_ok(c))throw PVZ_DENIED;
        D3D11_BOX box{static_cast<UINT>(x),static_cast<UINT>(y),0,static_cast<UINT>(x+width),static_cast<UINT>(y+height),1};c.pending=frame;c.pending_stage=staging;c.poisoned=true;
        c.dc->CopySubresourceRegion(staging.get(),0,0,0,0,texture.get(),0,&box);c.dc->End(query.get());c.dc->Flush();
        uint64_t gpu_wait_end=GetTickCount64()+2500; for(;;){if(!time_ok(c)||GetTickCount64()>=gpu_wait_end)throw PVZ_UNSAFE_PENDING;BOOL done=FALSE;HRESULT hr=c.dc->GetData(query.get(),&done,sizeof done,D3D11_ASYNC_GETDATA_DONOTFLUSH);if(hr==S_OK&&done)break;if(hr!=S_FALSE)winrt::check_hresult(hr);Sleep(1);}
        c.poisoned=false;c.pending=nullptr;c.pending_stage=nullptr;
        D3D11_MAPPED_SUBRESOURCE map{};HRESULT hr;uint64_t map_wait_end=GetTickCount64()+1500; do{if(!time_ok(c)||GetTickCount64()>=map_wait_end)throw PVZ_DENIED;hr=c.dc->Map(staging.get(),0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&map);if(hr==DXGI_ERROR_WAS_STILL_DRAWING)Sleep(1);}while(hr==DXGI_ERROR_WAS_STILL_DRAWING);winrt::check_hresult(hr);
        struct Unmap{ID3D11DeviceContext* d;ID3D11Texture2D* s;~Unmap(){d->Unmap(s,0);}} unmap{c.dc.get(),staging.get()};
        if(!map.pData||map.RowPitch<width*4||!alive(c)||!reverse(c)||!time_ok(c))throw PVZ_DENIED;
        for(uint32_t row=0;row<height;++row){std::memcpy(output+uint64_t(row)*width*4,static_cast<uint8_t const*>(map.pData)+uint64_t(row)*map.RowPitch,width*4);}
        if(!pvz_wgc_compute_alpha_diag(output,width,height,static_cast<uint32_t>(x),static_cast<uint32_t>(y),&result.alpha_nonopaque,&result.alpha_diag))throw PVZ_GEOMETRY;
        auto end_qpc=qpc100();if(!alive(c)||!reverse(c)||!time_ok(c))throw PVZ_DENIED;
        if(uint32_t end_ts_code=pvz_wgc_classify_end_ts(static_cast<uint64_t>(timestamp),end_qpc);end_ts_code!=PVZ_TS_OK){result.ts_reject_code=end_ts_code;throw PVZ_DENIED;}
        result.identity_after=1;
        result.client_w=width;result.client_h=height;result.stride=width*4;result.bytes_written=bytes;result.frame_qpc_100ns=static_cast<uint64_t>(timestamp);
        result.client_left=c.r.client_left;result.client_top=c.r.client_top;result.client_right=c.r.client_right;result.client_bottom=c.r.client_bottom;
        result.dwm_left=c.r.dwm_left;result.dwm_top=c.r.dwm_top;result.dwm_right=c.r.dwm_right;result.dwm_bottom=c.r.dwm_bottom;c.last_qpc=result.frame_qpc_100ns;status=PVZ_OK;
    }catch(winrt::hresult_error const& e){status=classify(e,&result);}catch(PvzWgcStatus s){status=s;}catch(...){status=PVZ_API_ERROR;}
    if(c.poisoned)status=PVZ_UNSAFE_PENDING;if(status!=PVZ_OK){std::memset(output,0,static_cast<size_t>(bytes));result.bytes_written=0;result.alpha_nonopaque=0;result.alpha_diag={};}
    if(frame&&!c.poisoned)try{frame.Close();}catch(...){c.cleanup_bad=true;std::memset(output,0,static_cast<size_t>(bytes));result.bytes_written=0;result.alpha_nonopaque=0;result.alpha_diag={};status=PVZ_CLEANUP_UNVERIFIED;}
    result.status=status;result.cleanup_state=c.poisoned?2:(c.cleanup_bad?3:0);*out=result;return status;
}
extern "C" int32_t __stdcall pvz_wgc_close(void* opaque){
    if(!opaque)return PVZ_DENIED;auto c=static_cast<Context*>(opaque);if(GetCurrentThreadId()!=c->thread)return PVZ_WRONG_THREAD;if(c->poisoned)return PVZ_UNSAFE_PENDING;
    bool ok=cleanup(*c);delete c;return ok?PVZ_OK:PVZ_CLEANUP_UNVERIFIED;
}
