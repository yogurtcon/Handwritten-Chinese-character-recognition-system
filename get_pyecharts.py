from pyecharts.charts import Line
from pyecharts import options as opts


def line_smooth(epochs_range, train_loss, val_loss, train_acc, val_acc) -> Line:
    c = (
        Line()
        .add_xaxis(epochs_range)
        .add_yaxis("train loss", train_loss, is_smooth=True)
        .add_yaxis("val loss", val_loss, is_smooth=True)
        .add_yaxis("train acc", train_acc, is_smooth=True)
        .add_yaxis("val acc", val_acc, is_smooth=True)
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                axistick_opts=opts.AxisTickOpts(is_align_with_label=True),
                is_scale=True,
                boundary_gap=False,
                name='epochs'
            ),
        )
    )
    c.render('acc_loss_pyecharts.html')
