"""Ridgeline PDFs with full-base circles and selected-gene sectors."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import sem, t
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Circle,Wedge
from matplotlib.offsetbox import DrawingArea,AnnotationBbox
from matplotlib.text import Text
from scipy.stats import gaussian_kde
from base_common import *

plt.rcParams.update({"font.family":"Arial","font.size":7,"axes.titlesize":8,"axes.labelsize":7,
    "axes.linewidth":.65,"axes.spines.top":False,"axes.spines.right":False,
    "pdf.fonttype":42,"svg.fonttype":"none","legend.frameon":False})
LABELS={"own_top100":"Own base Top-100","signature_anchor":"Shared SIGnature base Top-100"}


def nice(dataset):
    return {"GSE140440":"GSE140440 / docetaxel", "GSE223779":"GSE223779 FA34 / crizotinib",
        "GSE149383_PC9_DTEP_sample_grouped":"PC9 / DTEP state",
        "GSE165897_pre_EOC":"GSE165897 / pretreatment EOC",
        "GSE223003_carboplatin_pre":"ReSisTrace / carboplatin",
        "GSE223003_olaparib_pre":"ReSisTrace / olaparib",
        "GSE223003_nk_pre":"ReSisTrace / NK killing"}.get(dataset,dataset)


def stratum_name(dataset,stratum):
    if stratum=="All":return "All labeled cells"
    positive=stratum=="label_1"
    if "PC9" in dataset:return "DTEP day 11" if positive else "Untreated day 0"
    if "_nk_" in dataset:return "Escape-prone" if positive else "Killing-sensitive"
    if "EOC" in dataset:return "Clinical R" if positive else "Clinical S"
    return "Resistant" if positive else "Sensitive"


def icons(ax,y,row,color):
    area=DrawingArea(42,22,0,0)
    radius=3.5
    area.add_artist(Circle((9,14),radius,facecolor=color,edgecolor=color,lw=.4))
    area.add_artist(Text(9,3,"100%",ha="center",va="center",fontsize=5.2))
    area.add_artist(Circle((32,14),radius,facecolor="#E8E8E8",edgecolor="#999999",lw=.35))
    if row.set_kind == "signature_anchor":
        fraction = row.n_measured / row.n_selected
        label = f"{int(row.n_measured)}/{int(row.n_selected)}"
    else:
        fraction = float(row.gene_percent) / 100
        label = f"{row.gene_percent:.2f}%"
    if fraction>0:area.add_artist(Wedge((32,14),radius,90-360*fraction,90,facecolor=color,edgecolor="none"))
    area.add_artist(Text(32,3,label,ha="center",va="center",fontsize=5.2))
    artist=AnnotationBbox(area,(.5,y),xycoords=ax.get_yaxis_transform(),frameon=False,box_alignment=(.5,.5))
    ax.add_artist(artist)


def transform(values,log):return np.log1p(values) if log else np.asarray(values)


def mean_ci(raw, mean, level):
    if len(raw) < 2:
        return None
    scale = sem(raw)
    bounds = t.interval(level, len(raw) - 1, loc=mean, scale=scale) if scale > 0 else (mean, mean)
    return [float(bound) for bound in bounds]


def export_panel(page, row, part, raw, valid, title, log):
    record = {key: row[key] for key in (
        "experiment", "domain", "population", "stratum", "set_kind", "method",
        "n_cells", "n_total_cells", "n_base_genes", "n_selected", "n_measured", "gene_percent"
    )}
    record.update(title=title, complete_reference=valid,
                  sector_fraction=float(row.n_measured / row.n_selected if row.set_kind == "signature_anchor"
                                        else row.gene_percent / 100))
    record["genes"] = [
        dict(gene_symbol=gene.gene_symbol, measured=bool(gene.measured),
             mean_normalized_abs_ig=float(gene.value) if np.isfinite(gene.value) else None)
        for gene in part.itertuples()
    ]
    record["mean"] = float(row["mean"]) if valid else None
    record["ci95"] = mean_ci(raw, record["mean"], .95) if valid else None
    record["ci99"] = mean_ci(raw, record["mean"], .99) if valid else None
    record["quantiles"] = dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                                  map(float, np.quantile(raw, [0,.05,.25,.50,.75,.95,1])))) if valid else None
    page["series"].append(record)


def figure_path(dest, stem):
    folder = dest / stem
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (stem + ".pdf")


def save_plot_data(path, pages, log):
    import json
    payload = dict(pdf=path.name, x_axis="log1p" if log else "linear",
                   units="Mean normalized absolute IG; total per cell is 1000",
                   ci="Student t interval across gene means; displayed lower bound clipped at zero",
                   pages=pages)
    path.with_name("plot_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False,
                   default=lambda value: value.item()) + "\n", encoding="utf-8")


def draw_panel(fig,slot,summary,values,experiment,domain,population,stratum,set_kind,xmax,log,show_names,title,page):
    grid=slot.subgridspec(1,2,width_ratios=[4.2,1.15],wspace=.04)
    ax=fig.add_subplot(grid[0]);coverage=fig.add_subplot(grid[1],sharey=ax)
    selected=summary[(summary.experiment==experiment)&(summary.domain==domain)&(summary.population==population)&
                     (summary.stratum==stratum)&(summary.set_kind==set_kind)]
    curves=[];pool=[];n_cells=0;n_total=0;base_n=None
    for i,method in enumerate(METHODS):
        records=selected[selected.method.eq(method)]
        if len(records)!=1:
            curves.append(None);continue
        row=records.iloc[0];n_cells=int(row.n_cells);n_total=int(row.n_total_cells);base_n=int(row.n_base_genes)
        part=values[(values.experiment==experiment)&(values.domain==domain)&(values.population==population)&
                    (values.stratum==stratum)&(values.set_kind==set_kind)&(values.method==method)]
        raw=part.value.to_numpy(float)
        valid=bool(row.comparable) and len(raw)==row.n_selected and np.isfinite(raw).all()
        export_panel(page,row,part,raw,valid,title,log)
        x=transform(raw,log) if valid else np.array([])
        curves.append((row,raw,x,valid));pool.extend(x[x>0].tolist())
    grid_x=np.linspace(0,float(transform(xmax,log)),500)
    pool=np.asarray(pool)
    spread=min(np.std(pool),np.subtract(*np.percentile(pool,[75,25]))/1.349) if len(pool)>1 else .1
    if not np.isfinite(spread) or spread<=1e-8:spread=max(np.std(pool),.02) if len(pool)>1 else .1
    bandwidth=max(.9*spread*max(len(pool),1)**(-.2),float(grid_x[-1])*.003,.0001)
    densities=[]
    for curve in curves:
        if curve is None or not curve[3]:densities.append(None);continue
        x=curve[2];positive=x[x>0]
        if len(positive)>2 and np.std(positive)>1e-10:
            kde=gaussian_kde(positive,bw_method=bandwidth/np.std(positive,ddof=1))
            densities.append((kde(grid_x)+kde(-grid_x))*len(positive)/len(x))
        else:densities.append(None)
    height=max([float(d.max()) for d in densities if d is not None]+[1e-9])
    for i,(method,curve,density) in enumerate(zip(METHODS,curves,densities)):
        y=3-i;color=COLORS[method]
        if curve is None:
            ax.text(.02,y,"Not available",transform=ax.get_yaxis_transform(),fontsize=6,va="center");continue
        row,raw,x,valid=curve;icons(coverage,y,row,color)
        if not valid:
            message="No labeled cells" if row.n_cells==0 else f"NA: {int(row.n_measured)}/{int(row.n_selected)} genes measured"
            ax.text(.02,y,message,transform=ax.get_yaxis_transform(),fontsize=5.8,va="center",color="#666666");continue
        if density is not None:
            ridge=density/height*.58
            ax.fill_between(grid_x,y,y+ridge,color=color,alpha=.7,linewidth=0)
            ax.plot(grid_x,y+ridge,color=color,lw=.45)
        else:
            for location in np.unique(x):
                ax.plot([location,location],[y,y+.45*np.mean(x==location)],color=color,lw=1)
        if density is not None and row.zero_genes:
            ax.plot([0,0],[y,y+.45*row.zero_genes/row.n_selected],color=color,lw=1,zorder=5)
        outer = mean_ci(raw,row["mean"],.99) or [row["mean"]]*2
        ax.plot(transform(np.maximum(outer,0),log),[y,y],color=color,lw=2.2,solid_capstyle="butt")
        light=np.array(matplotlib.colors.to_rgb(color))*.55+.45
        inner = mean_ci(raw,row["mean"],.95) or [row["mean"]]*2
        ax.plot(transform(np.maximum(inner,0),log),[y,y],color=light,lw=3.5,solid_capstyle="butt")
        ax.plot(float(transform(row["mean"],log)),y,marker="o" if method=="signature" else "^",markersize=4,
                markerfacecolor="white",markeredgecolor=color,markeredgewidth=.9,zorder=5)
        if row.zero_genes:ax.text(.98,y+.28,f"0: {int(row.zero_genes)}/{int(row.n_selected)}",transform=ax.get_yaxis_transform(),ha="right",fontsize=4.7,color="#555555")
    ax.set_ylim(-.45,3.8);ax.set_xlim(0,float(transform(xmax,log)))
    ax.set_yticks([3,2,1,0],[NAMES[m] for m in METHODS] if show_names else [""]*4)
    ax.tick_params(axis="both",labelsize=6,length=2.5,pad=2)
    if log:
        candidates=np.array([0,.5,1,2,5,10,20,50,100,200,500,1000]);ticks=candidates[candidates<=xmax]
        if len(ticks)>6:ticks=ticks[np.linspace(0,len(ticks)-1,6).astype(int)]
        ax.set_xticks(np.log1p(ticks),[f"{x:g}" for x in ticks])
    ax.set_xlabel("Mean normalized |IG|"+(" (log1p axis)" if log else ""),fontsize=6)
    ax.set_title(title+f"\ncells={n_cells:,}"+(f"/{n_total:,}" if n_cells!=n_total else ""),fontsize=7)
    coverage.set_ylim(-.45,3.8);coverage.axis("off")
    header = "SIGnature\nreference" if set_kind == "signature_anchor" else f"Base genes\n{base_n:,}" if base_n is not None else ""
    if header:coverage.text(.5,1.01,header,ha="center",va="bottom",fontsize=5.2,transform=coverage.transAxes)
    return ax


def finish(fig,title,set_kind):
    height=fig.get_figheight()
    fig.suptitle(title,fontsize=10,y=1-.04/height)
    handles=[Line2D([0],[0],color="#777777",lw=2.2,label="99% mean CI"),
             Line2D([0],[0],color="#BBBBBB",lw=3.5,label="95% mean CI"),
             Line2D([0],[0],marker="o",color="none",markerfacecolor="white",markeredgecolor="#555555",label="SIGnature mean"),
             Line2D([0],[0],marker="^",color="none",markerfacecolor="white",markeredgecolor="#555555",label="Classifier mean")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.16/height),ncol=4,fontsize=6,handlelength=1.5,columnspacing=1)
    if set_kind == "signature_anchor":
        caption = "Circle: fixed source SIGnature Top 100; sector: measured and graph-eligible genes / reference genes."
    else:
        caption = "Circle: common eligible base genes; sector: selected / base gene count."
    fig.text(.5,.055/height,caption+" CIs use gene-level means.",ha="center",fontsize=5.5)
    fig.subplots_adjust(left=.09,right=.985,top=1-.72/height,bottom=.83/height,wspace=.28,hspace=.55)


def base_figures(summary,values,dest,set_kind,log):
    data=summary[(summary.domain=="base")&(summary.population=="cohort")&(summary.set_kind==set_kind)]
    suffix="log1p" if log else "linear"
    path=figure_path(dest,"01_base_"+set_kind+"_"+suffix)
    pages=[]
    with PdfPages(path) as pdf:
        for dataset in dict.fromkeys(data.dataset):
            part=data[data.dataset.eq(dataset)];exps=list(dict.fromkeys(part.experiment));rows=len(exps)
            raw=values[values.experiment.isin(exps)&values.domain.eq("base")&values.population.eq("cohort")].value.dropna()
            xmax=max(float(raw.max())*1.05,1.)
            fig=plt.figure(figsize=(9.2,2.0*rows+1.4));grid=fig.add_gridspec(rows,3)
            page=dict(page=len(pages)+1,dataset=dataset,x_max=xmax,series=[])
            pages.append(page)
            for i,e in enumerate(exps):
                fold=part.loc[part.experiment.eq(e),"fold"].iloc[0]
                for j,stratum in enumerate(["All","label_0","label_1"]):
                    draw_panel(fig,grid[i,j],summary,values,e,"base","cohort",stratum,set_kind,xmax,log,j==0,
                               fold+" / "+stratum_name(dataset,stratum),page)
            finish(fig,nice(dataset)+" — "+LABELS[set_kind]+"\nComplete base: train + validation + test",set_kind)
            pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
    save_plot_data(path,pages,log)
    return path


def transfer_cases(summary):
    cases=[]
    for (e,domain),part in summary[summary.domain.ne("base")].groupby(["experiment","domain"],sort=False):
        target=part.target_dataset.iloc[0];pops=set(part.population)
        if target=="GSE223779_TS485T":
            for population in ["crizotinib","alectinib"]:cases.append(("primary",e,domain,population,target))
        elif target=="GSE154600_EPI_external":cases.append(("primary",e,domain,"cohort",target))
        elif target=="GSE140440":
            for population in ["DU145","PC3"]:cases.append(("pressure",e,domain,population,target))
        elif target=="GSE163836":
            for population in ["FCIBC02","SUM149"]:cases.append(("pressure",e,domain,population,target))
        elif target=="GSE202234":
            for population in sorted(pops-{"cohort"}):cases.append(("pressure",e,domain,population,target))
    return cases


def transfer_figures(summary,values,dest,set_kind,log):
    paths=[];suffix="log1p" if log else "linear"
    cases=transfer_cases(summary)
    for tier in ["primary","pressure"]:
        selected=[c for c in cases if c[0]==tier]
        if not selected:continue
        path=figure_path(dest,"02_"+tier+"_transfer_"+set_kind+"_"+suffix)
        pages=[]
        with PdfPages(path) as pdf:
            for _,e,domain,population,target in selected:
                source=summary.loc[summary.experiment.eq(e),"dataset"].iloc[0]
                raw=values[(values.experiment==e)&values.domain.isin(["base",domain])].value.dropna()
                xmax=max(float(raw.max())*1.05,1.)
                fig=plt.figure(figsize=(8.0,7.2));grid=fig.add_gridspec(3,2)
                page=dict(page=len(pages)+1,dataset=target,population=population,x_max=xmax,series=[])
                pages.append(page)
                for i,stratum in enumerate(["All","label_0","label_1"]):
                    draw_panel(fig,grid[i,0],summary,values,e,"base","cohort",stratum,set_kind,xmax,log,True,
                               "Base / "+stratum_name(source,stratum),page)
                    draw_panel(fig,grid[i,1],summary,values,e,domain,population,stratum,set_kind,xmax,log,False,
                               "Target / "+stratum_name(target,stratum),page)
                finish(fig,e+" → "+target+" / "+population+"\n"+LABELS[set_kind]+"; source genes and source K frozen",set_kind)
                pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
        save_plot_data(path,pages,log)
        paths.append(path)
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--save_path",type=Path,default=OUT/"figures")
    p.add_argument("--datasets",nargs="+");p.add_argument("--stage",choices=["base","all"],default="all")
    a=p.parse_args();a.save_path.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(OUT/"source_data/ridge_summary.tsv",sep="\t")
    values=pd.read_csv(OUT/"source_data/ridge_gene_values.tsv.gz",sep="\t")
    if a.datasets:
        summary=summary[summary.dataset.isin(a.datasets)];values=values[values.experiment.isin(summary.experiment)]
    files=[]
    for kind in LABELS:
        for log in [False,True]:
            files.append(base_figures(summary,values,a.save_path,kind,log))
            if a.stage=="all":files.extend(transfer_figures(summary,values,a.save_path,kind,log))
    for path in files:print("PDF",path,flush=True)


if __name__=="__main__":main()
