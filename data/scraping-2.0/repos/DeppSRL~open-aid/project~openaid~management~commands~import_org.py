# coding=utf-8
from optparse import make_option
import csvkit
from os.path import join
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.core.management.base import CommandError, BaseCommand
from django.db.models import Sum
from openaid.projects.models import Organization, AnnualFunds



class Command(BaseCommand):
    args = '<crs_file crs_file ...>'
    help = 'Importa le informazioni sui fondi multilaterali %s' % join(settings.RESOURCES_PATH, 'multilateral')

    option_list = BaseCommand.option_list + (
        make_option(
            "-f",
            "--file",
            dest = "organizations",
            help = "specify import file",
            metavar = "FILE",
            default=join(settings.RESOURCES_PATH, 'multilateral/organizations.csv'),
        ),
        make_option(
            "--funds",
            dest = "funds",
            help = "specify funds import file",
            metavar = "FILE",
            default=join(settings.RESOURCES_PATH, 'multilateral/funds.csv'),
        ),
        make_option('-c', '--clean',
            action='store_true', dest='clean', default=False,
            help="Clean old organizations and annual funds before executing the import."),
        make_option('-o', '--override',
            action='store_true', dest='override',
            help="Override old values."),
    )

    def delete_organizations(self):
        answer = raw_input('Are you sure? (Yes/No)')
        if answer.lower() in ('yes', 'y'):
            self.stdout.write(u'Deleting %s annual funds' % AnnualFunds.objects.count())
            AnnualFunds.objects.all().delete()
            self.stdout.write(u'Deleting %s organizations' % Organization.objects.count())
            Organization.objects.all().delete()
            return True
        return False

    def handle(self, *args, **options):

        self.stdout.write(u'Import organizations and annual funds')

        if options.get('clean') and not self.delete_organizations():
            raise CommandError("Import aborted")

        orgs = {}

        self.stdout.write(u'Read: %s' % options['organizations'])

        for row in csvkit.DictReader(open(options['organizations'])):
            self.stdout.write(u'Try to retrive or create: %s' % row)

            parent = None
            parent_acronym = row['parent_acronym'].strip()
            if parent_acronym:
                try:
                    parent = Organization.objects.get(acronym=parent_acronym)
                except ObjectDoesNotExist:
                    raise Exception("acronym:{} not in db".format(parent_acronym))

            order = row['order'].strip()
            try:
                order = int(order)
            except ValueError:
                order = 0

            org, created = Organization.objects.get_or_create(
                acronym=row['acronym'].strip(),
                defaults={
                    'name': row['name_en'].strip(),
                    'parent': parent,
                    'order': order
                    }
            )
            if created:
                self.stdout.write(u'Create new organization: %s' % org)
            else:
                self.stdout.write(u'Already created organization: %s' % org)
            orgs[org.acronym] = org

        for row in csvkit.DictReader(open(options['funds'])):

            commitment = float(row['commitment'].replace(',', '.') or 0.0)
            disbursement = float(row['disbursement'].replace(',', '.') or 0.0)

            fund, created = AnnualFunds.objects.get_or_create(
                organization=orgs[row['acronym'].strip()],
                year=row['year'],
                defaults={
                    'commitment': commitment,
                    'disbursement': disbursement,
                }
            )
            if created:
                self.stdout.write(u'Add fund for %s' % fund)
            elif options['override']:
                to_save = False
                if fund.commitment != commitment:
                    self.stdout.write(u'Update fund commitment %s for %s' % (commitment, fund))
                    fund.commitment = commitment
                    to_save = True
                if fund.disbursement != disbursement:
                    self.stdout.write(u'Update fund disbursement %s for %s' % (disbursement, fund))
                    fund.disbursement = disbursement
                    to_save = True

                if to_save:
                    fund.save()
                    self.stdout.write(u'Updated fund for %s' % fund)


        # check the data
        self.stdout.write("*********** CHECK THE DATA INSERTED ***********")
        for main_org in Organization.objects.filter(parent__isnull=True).order_by('name'):

            if Organization.objects.filter(parent=main_org).count() == 0:
                continue

            for fund in AnnualFunds.objects.filter(organization=main_org).order_by('year'):
                if AnnualFunds.objects.filter(organization__parent=main_org, year=fund.year).count() ==0:
                    self.stdout.write(u'For org:{} year:{}, there are no children to compare with, skip'.format(main_org.name, fund.year,))
                    continue
                children_sum = AnnualFunds.objects.filter(organization__parent=main_org, year=fund.year).aggregate(**{'commitment':Sum('commitment'),'disbursement':Sum('disbursement')})

                if fund.commitment != children_sum['commitment']:
                    self.stdout.write(u'Error with Commit sums for org:{} year:{}, total:{}, sum of children:{}'.format(main_org.name, fund.year, fund.commitment, children_sum['commitment'] ))

                if fund.disbursement != children_sum['disbursement']:
                    self.stdout.write(u'Error with Disbur sums for org:{} year:{}, total:{}, sum of children:{}'.format(main_org.name, fund.year, fund.disbursement, children_sum['disbursement'] ))
        self.stdout.write("*********** DONE ***********")
